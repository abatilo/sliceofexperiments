# syntax=docker/dockerfile:1.4
FROM golang:1.20.4 as go-modules

# Install dependencies
WORKDIR /app
COPY ./go.mod ./go.sum ./
RUN \
  go mod download -x

FROM go-modules as go-builder

COPY ./cmd ./cmd
RUN \
  CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go install -ldflags="-w -s" ./cmd/...
