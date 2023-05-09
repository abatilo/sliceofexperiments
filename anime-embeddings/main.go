package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"time"

	"github.com/cohere-ai/cohere-go"
	"github.com/nstratos/go-myanimelist/mal"
)

func main() {
	var cohereAPIKey string
	flag.StringVar(&cohereAPIKey, "cohere-api-key", "", "Cohere API key")
	flag.Parse()

	// Read a file named anime.jsonl
	// Create a slice of map[string]interface{}
	// Read each line one at a time
	// Decode each line into a map[string]interface{}
	// Append the map to the slice
	// Print the slice
	f, err := os.Open("anime.jsonl")
	if err != nil {
		panic(err)
	}
	defer f.Close()

	var anime []map[string]interface{}

	dec := json.NewDecoder(f)
	for dec.More() {
		var m map[string]interface{}
		err := dec.Decode(&m)
		if err != nil {
			panic(err)
		}

		// Remove nil and empty synopsis
		if m["synopsis"] == nil || m["synopsis"] == "" {
			continue
		}

		var reducedM map[string]interface{}
		reducedM = make(map[string]interface{})

		reducedM["id"] = m["id"].(float64)
		reducedM["title"] = m["title"].(string)
		reducedM["alternative_titles"] = m["alternative_titles"].(map[string]interface{})
		reducedM["synopsis"] = m["synopsis"].(string)

		anime = append(anime, reducedM)
	}

	co, err := cohere.CreateClient(cohereAPIKey)
	if err != nil {
		fmt.Println(err)
		return
	}

	i := 0
	batchSize := 96
	for i < len(anime) {
		remaining := len(anime) - i
		if remaining > batchSize {
			remaining = batchSize
		}
		animeSlice := anime[i : i+remaining]
		fmt.Printf("Sending batch %d-%d\n", i, i+remaining)

		synopses := make([]string, len(animeSlice))
		for k, a := range animeSlice {
			synopses[k] = a["synopsis"].(string)
		}

		resp, err := co.Embed(cohere.EmbedOptions{
			Model: "embed-multilingual-v2.0",
			Texts: synopses,
		})
		if err != nil {
			panic(err)
		}

		j := i
		for _, embedding := range resp.Embeddings {
			anime[j]["embedding"] = embedding
			j++
		}
		i += batchSize
	}

	// Write the slice to a file
	b, err := json.Marshal(anime)
	if err != nil {
		panic(err)
	}

	f, err = os.Create("anime-embeddings.json")
	if err != nil {
		panic(err)
	}
	defer f.Close()

	_, err = f.Write(b)
	if err != nil {
		panic(err)
	}
}

type clientIDTransport struct {
	Transport http.RoundTripper
	ClientID  string
}

func (c *clientIDTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if c.Transport == nil {
		c.Transport = http.DefaultTransport
	}
	req.Header.Add("X-MAL-CLIENT-ID", c.ClientID)
	return c.Transport.RoundTrip(req)
}

func download() {
	var clientID string
	flag.StringVar(&clientID, "client-id", "", "MyAnimeList client ID")
	flag.Parse()

	publicInfoClient := &http.Client{
		// Create client ID from https://myanimelist.net/apiconfig.
		Transport: &clientIDTransport{ClientID: clientID},
	}

	ctx := context.Background()
	c := mal.NewClient(publicInfoClient)

	// create a file to write all anime data to
	// delete the file if it already exists
	_ = os.Remove("anime.jsonl")
	f, err := os.Create("anime.jsonl")
	if err != nil {
		panic(err)
	}
	defer f.Close()

	retries := 0
	failures := 0

	for i := 0; i < 100000; i++ {
		time.Sleep(1 * time.Second)

		anime, resp, err := c.Anime.Details(ctx, i,
			mal.Fields{
				"alternative_titles",
				"synopsis",
			},
		)
		if resp.StatusCode == http.StatusNotFound {
			fmt.Printf("No anime found with ID %d\n", i)
			continue
		}

		if resp.StatusCode != http.StatusOK || err != nil {
			fmt.Printf("status code: %d; ", resp.StatusCode)
			fmt.Printf("error: %v\n", err)

			failures = failures + 1
			i = i - 1
			retries = retries + 1

			if retries > 3 {
				time.Sleep(1 * time.Minute)
				continue
			}
		}

		retries = 0

		fmt.Printf("%#v\n", i)

		// serialize anime to json and append it to f
		b, err := json.Marshal(anime)
		if err != nil {
			panic(err)
		}
		_, err = f.Write(b)
		if err != nil {
			panic(err)
		}
		_, err = f.WriteString("\n")
		if err != nil {
			panic(err)
		}
	}
}
