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

	"github.com/dgraph-io/badger/v3"
	"github.com/sashabaranov/go-openai"
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

			if q.CorrectAnswer == guess {
				fmt.Printf("Correct.")
			} else {
				fmt.Printf("Incorrect.")
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
	apiKey := os.Getenv("OPENAI_API_KEY")
	client := openai.NewClient(apiKey)

	questionTemplate := fmt.Sprintf(
		`Question:
%s

Possible answers:
%s

`,
		q.Question,
		strings.Join(q.PossibleAnswers, "\n"),
	)

	var resp openai.ChatCompletionResponse
	var err error

	// retry up to 30 times
	attempts := 0
	for attempts == 0 || err != nil {
		if attempts > 30 {
			panic(err)
		}
		resp, err = client.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
			Temperature: 0,
			Model:       openai.GPT4,
			Messages: []openai.ChatCompletionMessage{
				{
					Role:    "system",
					Content: `You are a knowledge assistant. I will ask you a multiple choice question and you will answer it.`,
				},
				{
					Role: "user",
					Content: `You will be given a question and then you will be presented with possible
answers to choose from. If you're not sure of the answer, make your
best guess and pick one of the answers.

Follow these instructions:
- Think out loud, step by step
- Insert two blank lines to separate your answer from your explanation
- Write one of your answers and write it exactly character for character as it appears in the list of possible answers
`,
				},
				{
					Role: "user",
					Content: `Question:
What language can Harry Potter speak?

Possible answers:
Goblin
English
Mermish
Parseltounge
`,
				},
				{
					Role: "assistant",
					Content: `Thinking out loud:
Harry never has a need to speak goblin, so I can eliminate that answer. English
is the known language that he speaks which leaves mermish and parseltounge.
Mermish would be the language that merpeople speak, so I can eliminate that
answer. That leaves parseltounge as the correct answer.

Answer:


Parseltounge`,
				},
				{
					Role:    "user",
					Content: questionTemplate,
				},
			},
		})
		attempts++

		if err != nil {
			// Sometimes the model is overloaded and returns an error. Wait a bit and try again.
			fmt.Printf("Error: %s. Sleeping before trying again...\n", err)
			time.Sleep(1 * time.Minute)
		}
	}

	fmt.Println("=========================================")
	fmt.Println()
	fmt.Println()
	fmt.Println(questionTemplate)
	fmt.Println(resp.Choices[0].Message.Content)

	// Split guess by line and take the last line
	guessLines := strings.Split(resp.Choices[0].Message.Content, "\n")
	guess := strings.TrimSpace(guessLines[len(guessLines)-1])

	return guess
}
