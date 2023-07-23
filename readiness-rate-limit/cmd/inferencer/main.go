package main

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	gosundheit "github.com/AppsFlyer/go-sundheit"
	"github.com/AppsFlyer/go-sundheit/checks"
	healthhttp "github.com/AppsFlyer/go-sundheit/http"
	"github.com/google/uuid"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

const (
	// MAX_BATCH_SIZE simulates the maximum number of items that can be fit in a
	// GPU batch
	MAX_BATCH_SIZE = 40

	// QUEUE_DEPTH_FAILURE_THRESHOLD is the depth at which it means we're not
	// keeping up with processing and we're getting overloaded
	QUEUE_DEPTH_FAILURE_THRESHOLD = 70

	// BATCH_EVERY is how often to ship a batch
	BATCH_EVERY = 8000 * time.Millisecond
)

type BatchItem struct {
	id uuid.UUID
}

func (b *BatchItem) Process() error {
	fmt.Println("Processing item: ", b.id)
	time.Sleep(time.Duration(rand.Intn(1000)) * time.Millisecond)
	return nil
}

// Queue is a dynamic sized queue that can be used to simulate a GPU queue
type Queue struct {
	Items []*BatchItem
	mu    sync.Mutex
}

func (q *Queue) Push(item *BatchItem) {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.Items = append(q.Items, item)
}

// PopBatch pops a batch of items from the queue
func (q *Queue) PopBatch() []*BatchItem {
	q.mu.Lock()
	defer q.mu.Unlock()
	numItems := int(math.Min(MAX_BATCH_SIZE, float64(len(q.Items))))
	batch := q.Items[:numItems]
	q.Items = q.Items[numItems:]
	return batch
}

type Batcher struct {
	Items          *Queue
	ReturnChannels map[uuid.UUID]chan error
	mu             sync.Mutex
}

// SendToBatch sends an item to the batcher and returns a channel that will
// return the result of the processing.
func (b *Batcher) SendToBatch(item *BatchItem) (chan error, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.ReturnChannels[item.id] = make(chan error)
	b.Items.Push(item)
	return b.ReturnChannels[item.id], nil
}

// ProcessBatch processes the batch
func (b *Batcher) ProcessBatch() {
	batch := b.Items.PopBatch()
	var wg sync.WaitGroup
	for _, item := range batch {
		wg.Add(1)
		go func(item *BatchItem) {
			b.mu.Lock()
			defer b.mu.Unlock()
			defer wg.Done()
			err := item.Process()
			ch := b.ReturnChannels[item.id]
			ch <- err
			delete(b.ReturnChannels, item.id)
			close(ch)
		}(item)
	}
	wg.Wait()
}

// Start starts the batcher which will call ProcessBatch when either there's
// more than MAX_BATCH_SIZE items in the queue or every 5 seconds, whichever
// comes first.
func (b *Batcher) Start() {
	ticker := time.NewTicker(BATCH_EVERY)
	go func() {
		for range ticker.C {
			b.ProcessBatch()
		}
	}()
}

func main() {
	batcher := &Batcher{
		Items:          &Queue{},
		ReturnChannels: make(map[uuid.UUID]chan error),
	}
	batcher.Start()

	numInFlight := &atomic.Int64{}

	queueMetric := promauto.NewGauge(prometheus.GaugeOpts{
		Name: "queue_depth",
		Help: "The depth of the queue",
	})
	queueMetric.Set(float64(numInFlight.Load()))

	h := gosundheit.New()
	_ = h.RegisterCheck(
		&checks.CustomCheck{
			CheckName: "batcher_queue_depth",
			CheckFunc: func(ctx context.Context) (interface{}, error) {
				inFlight := numInFlight.Load()
				if QUEUE_DEPTH_FAILURE_THRESHOLD < inFlight {
					return nil, fmt.Errorf("queue depth too high: %d", inFlight)
				}
				return inFlight, nil
			},
		},
		gosundheit.ExecutionPeriod(3*time.Second),
	)

	mux := http.NewServeMux()
	mux.HandleFunc("/push", func(w http.ResponseWriter, r *http.Request) {
		numInFlight.Add(1)
		queueMetric.Set(float64(numInFlight.Load()))
		defer func() {
			numInFlight.Add(-1)
			queueMetric.Set(float64(numInFlight.Load()))
		}()

		item := &BatchItem{
			id: uuid.New(),
		}
		ch, err := batcher.SendToBatch(item)
		<-ch

		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
	})
	mux.Handle("/readyz", healthhttp.HandleHealthJSON(h))
	mux.Handle("/metrics", promhttp.Handler())
	srv := http.Server{
		Addr:    ":8080",
		Handler: mux,
	}

	fmt.Println("Listening...")
	_ = srv.ListenAndServe()
}
