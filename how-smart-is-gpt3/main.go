package main

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/PullRequestInc/go-gpt3"
	"github.com/dgraph-io/badger/v3"
	"golang.org/x/time/rate"
)

type Question struct {
	Question        string
	CorrectAnswer   string
	PossibleAnswers []string
}

type Stats struct {
	TotalCorrect      int
	TotalQuestions    int
	CorrectPercentage float32
}

func main() {
	db, err := badger.Open(badger.DefaultOptions("results"))
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	categories := []string{
		"animals",
		"brain-teasers",
		"celebrities",
		"entertainment",
		"for-kids",
		"general",
		"geography",
		"history",
		"hobbies",
		"humanities",
		"literature",
		"movies",
		"music",
		"newest",
		"people",
		"rated",
		"religion-faith",
		"science-technology",
		"sports",
		"television",
		"video-games",
		"world",
	}

	questionsByCategory := map[string]Stats{}

	// limit requests to 3000 per minute == 50 per second
	// https://help.openai.com/en/articles/5955598-is-api-usage-subject-to-any-rate-limits
	limiter := rate.NewLimiter(50, 50)
	completed := 0

	totalOverallCorrect := 0
	totalOverallQuestions := 0

	for _, category := range categories {
		url := fmt.Sprintf(
			"https://raw.githubusercontent.com/uberspot/OpenTriviaQA/master/categories/%s",
			category,
		)
		resp, err := http.Get(url)
		if err != nil {
			panic(err)
		}
		defer resp.Body.Close()

		allQuestions := ParseQuestions(resp.Body)

		numCorrect := 0
		numTotal := 0

		for _, q := range allQuestions {
			limiter.Wait(context.Background())

			var guess string

			// If we've already answered this question, use the cached answer
			err := db.View(func(txn *badger.Txn) error {
				item, err := txn.Get([]byte(q.Question))
				if err != nil {
					return err
				}

				return item.Value(func(val []byte) error {
					guess = string(val)
					return nil
				})
			})
			if err != nil {
				guess = AskGPT3Question(q)
			}

			if q.CorrectAnswer == guess {
				numCorrect++
			}
			numTotal++

			fmt.Printf("%s was ", q.Question)
			if q.CorrectAnswer == guess {
				fmt.Printf("correct.")
			} else {
				fmt.Printf("incorrect.")
			}
			fmt.Printf(
				" You guessed \"%s\". The correct answer was \"%s\".\n",
				guess,
				q.CorrectAnswer,
			)

			db.Update(func(txn *badger.Txn) error {
				err := txn.Set([]byte(q.Question), []byte(guess))
				return err
			})

			completed++
			if completed%1000 == 0 {
				log.Printf("Completed %d questions\n", completed)
			}
		}

		questionsByCategory[category] = Stats{
			TotalCorrect:      numCorrect,
			TotalQuestions:    numTotal,
			CorrectPercentage: float32(numCorrect) / float32(numTotal),
		}

		totalOverallCorrect += numCorrect
		totalOverallQuestions += numTotal
	}

	fmt.Println("Category\tCorrect\tTotal\tPercentage")
	for _, category := range categories {
		fmt.Printf(
			"%s\t%d\t%d\t%f\n",
			category,
			questionsByCategory[category].TotalCorrect,
			questionsByCategory[category].TotalQuestions,
			questionsByCategory[category].CorrectPercentage,
		)
	}

	fmt.Printf(
		"overall\t%d\t%d\t%f\n",
		totalOverallCorrect,
		totalOverallQuestions,
		float32(totalOverallCorrect)/float32(totalOverallQuestions),
	)
}

func ParseQuestions(rawText io.Reader) []Question {
	scanner := bufio.NewScanner(rawText)

	scanner.Split(bufio.ScanLines)

	allQuestions := []Question{}

	question := ""
	correctAnswer := ""
	possibleAnswers := []string{}
	for scanner.Scan() {
		line := scanner.Text()
		if len(line) == 0 {
			if question != "" && correctAnswer != "" && len(possibleAnswers) > 0 {
				allQuestions = append(allQuestions, Question{
					Question:        question,
					CorrectAnswer:   correctAnswer,
					PossibleAnswers: possibleAnswers,
				})
				question = ""
				correctAnswer = ""
				possibleAnswers = []string{}
			}
		} else if len(line) < 3 {
			continue
		} else if strings.HasPrefix(line, "#Q") {
			question = line[3:]
		} else if line[0] == '^' {
			correctAnswer = line[2:]
		} else {
			possibleAnswers = append(possibleAnswers, line[2:])
		}
	}

	if question != "" && correctAnswer != "" && len(possibleAnswers) > 0 {
		allQuestions = append(allQuestions, Question{
			Question:        question,
			CorrectAnswer:   correctAnswer,
			PossibleAnswers: possibleAnswers,
		})
	}

	return allQuestions
}

func AskGPT3Question(q Question) string {
	apiKey := os.Getenv("OPENAI_SECRET_KEY")
	client := gpt3.NewClient(
		apiKey,
		gpt3.WithDefaultEngine(gpt3.TextDavinci003Engine),
		gpt3.WithTimeout(10*time.Minute),
	)

	questionTemplate := fmt.Sprintf(
		`I am a highly intelligent multiple choice trivia bot. You are given a multiple choice question. You must choose the correct answer from one of answers. Only include the answer and nothing else.
Question:
%s

Possible answers:
%s

Your answer:`,
		q.Question,
		strings.Join(q.PossibleAnswers, "\n"),
	)

	var resp *gpt3.CompletionResponse
	var err error

	// retry up to 30 times
	attempts := 0
	for attempts == 0 || err != nil {
		if attempts > 30 {
			panic(err)
		}
		resp, err = client.Completion(context.Background(), gpt3.CompletionRequest{
			Prompt:           []string{questionTemplate},
			Temperature:      gpt3.Float32Ptr(0.0),
			MaxTokens:        gpt3.IntPtr(100),
			TopP:             gpt3.Float32Ptr(1.0),
			FrequencyPenalty: 0.0,
			PresencePenalty:  0.0,
			N:                gpt3.IntPtr(1),
		})
		attempts++

		if err != nil {
			// Sometimes the model is overloaded and returns an error. Wait a bit and try again.
			fmt.Printf("Error: %s. Sleeping before trying again...\n", err)
			time.Sleep(1 * time.Minute)
		}
	}

	guess := strings.TrimSpace(resp.Choices[0].Text)
	return guess
}
