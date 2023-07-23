package main

import (
	"fmt"
	"net/http"
	"net/http/httputil"
	"net/url"
)

func main() {
	u, _ := url.Parse("http://inferencer")
	p := httputil.NewSingleHostReverseProxy(u)
	fmt.Println("Listening on port 8080")
	_ = http.ListenAndServe(":8080", p)
}
