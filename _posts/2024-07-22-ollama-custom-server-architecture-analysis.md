---
layout: post
title: "Ollama Custom 서버 아키텍처 심층 분석 - Gin 기반 HTTP API 서버"
date: 2024-07-22
categories: [AI, Go, Server]
tags: [Ollama, Gin, HTTP Server, RESTful API, Go Web Framework]
---

# Ollama Custom 서버 아키텍처 심층 분석

Ollama Custom의 서버 아키텍처는 Gin 웹 프레임워크를 기반으로 구축된 고성능 HTTP API 서버입니다. 이 포스트에서는 `server/routes.go`의 1,700여 라인에 걸친 복잡한 서버 구현과 스케줄링 시스템을 상세히 분석해보겠습니다.

## 1. 서버 아키텍처 개요

### 1.1 핵심 구조체와 설계

```go
type Server struct {
    addr  net.Addr
    sched *Scheduler
}

type Scheduler struct {
    pendingReqCh  chan *LlmRequest  // 대기 중인 요청
    finishedReqCh chan *LlmRequest  // 완료된 요청
    expiredCh     chan *runnerRef   // 만료된 러너
    unloadedCh    chan any          // 언로드된 모델
    
    loaded   map[string]*runnerRef   // 로드된 모델 맵
    loadedMu sync.Mutex              // 동시성 제어
    
    loadFn       func(req *LlmRequest, f *ggml.GGML, gpus discover.GpuInfoList, numParallel int)
    newServerFn  func(gpus discover.GpuInfoList, model string, ...) (llm.LlamaServer, error)
    getGpuFn     func() discover.GpuInfoList
    getCpuFn     func() discover.GpuInfoList
    reschedDelay time.Duration
}
```

**아키텍처 특징:**
- **채널 기반 통신**: 비동기 요청 처리를 위한 채널 시스템
- **스케줄링 분리**: 모델 로딩과 요청 처리의 명확한 분리
- **함수형 인터페이스**: 테스트 가능한 의존성 주입 패턴
- **동시성 안전**: 뮤텍스를 활용한 안전한 상태 관리

### 1.2 Gin 라우터 설정

```go
func (s *Server) GenerateRoutes(rc *ollama.Registry) (http.Handler, error) {
    // CORS 설정
    corsConfig := cors.DefaultConfig()
    corsConfig.AllowWildcard = true
    corsConfig.AllowBrowserExtensions = true
    corsConfig.AllowHeaders = []string{
        "Authorization", "Content-Type", "User-Agent", "Accept",
        "X-Requested-With",
        // OpenAI 호환성 헤더들
        "OpenAI-Beta", "x-stainless-arch", "x-stainless-async",
        "x-stainless-custom-poll-interval", "x-stainless-helper-method",
        // ... 추가 헤더들
    }
    corsConfig.AllowOrigins = envconfig.AllowedOrigins()
    
    r := gin.Default()
    r.HandleMethodNotAllowed = true
    r.Use(
        cors.New(corsConfig),
        allowedHostsMiddleware(s.addr),
    )
    
    return r, nil
}
```

**CORS 및 보안:**
- **광범위한 CORS 지원**: 브라우저 확장 및 외부 애플리케이션 지원
- **OpenAI 호환성**: OpenAI API와 호환되는 헤더 지원
- **호스트 검증**: 허용된 호스트만 접근 가능하도록 제한
- **메소드 제어**: 허용되지 않은 HTTP 메소드 처리

## 2. RESTful API 엔드포인트 분석

### 2.1 핵심 API 라우팅 구조

```go
// 일반적인 엔드포인트
r.HEAD("/", func(c *gin.Context) { c.String(http.StatusOK, "Ollama is running") })
r.GET("/", func(c *gin.Context) { c.String(http.StatusOK, "Ollama is running") })
r.GET("/api/version", func(c *gin.Context) { 
    c.JSON(http.StatusOK, gin.H{"version": version.Version}) 
})

// 모델 관리
r.POST("/api/pull", s.PullHandler)        // 모델 다운로드
r.POST("/api/push", s.PushHandler)        // 모델 업로드
r.GET("/api/tags", s.ListHandler)         // 모델 목록
r.POST("/api/show", s.ShowHandler)        // 모델 정보
r.DELETE("/api/delete", s.DeleteHandler)  // 모델 삭제

// 모델 생성 및 관리
r.POST("/api/create", s.CreateHandler)    // 모델 생성
r.POST("/api/blobs/:digest", s.CreateBlobHandler)  // 블롭 생성
r.HEAD("/api/blobs/:digest", s.HeadBlobHandler)    // 블롭 확인
r.POST("/api/copy", s.CopyHandler)        // 모델 복사

// 추론 엔드포인트
r.GET("/api/ps", s.PsHandler)             // 실행 중인 모델
r.POST("/api/generate", s.GenerateHandler) // 텍스트 생성
r.POST("/api/chat", s.ChatHandler)        // 대화형 채팅
r.POST("/api/embed", s.EmbedHandler)      // 임베딩 생성
r.POST("/api/embeddings", s.EmbeddingsHandler) // 레거시 임베딩

// OpenAI 호환성
r.POST("/v1/chat/completions", openai.ChatMiddleware(), s.ChatHandler)
r.POST("/v1/completions", openai.CompletionsMiddleware(), s.GenerateHandler)
r.POST("/v1/embeddings", openai.EmbeddingsMiddleware(), s.EmbedHandler)
r.GET("/v1/models", openai.ListMiddleware(), s.ListHandler)
r.GET("/v1/models/:model", openai.RetrieveMiddleware(), s.ShowHandler)
```

**API 설계 원칙:**
- **RESTful 패턴**: HTTP 메소드와 리소스 기반 URL 설계
- **OpenAI 호환성**: 기존 OpenAI 클라이언트와 호환
- **미들웨어 활용**: 요청 변환과 검증을 위한 미들웨어 패턴
- **버전 관리**: `/v1/` 프리픽스로 API 버전 관리

### 2.2 GenerateHandler - 텍스트 생성 핵심

```go
func (s *Server) GenerateHandler(c *gin.Context) {
    checkpointStart := time.Now()
    var req api.GenerateRequest
    if err := c.ShouldBindJSON(&req); errors.Is(err, io.EOF) {
        c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
        return
    } else if err != nil {
        c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
        return
    }

    name := model.ParseName(req.Model)
    if !name.IsValid() {
        c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Model)})
        return
    }

    // 모델 언로드 요청 처리
    if req.Prompt == "" && req.KeepAlive != nil && int(req.KeepAlive.Seconds()) == 0 {
        s.sched.expireRunner(m)
        c.JSON(http.StatusOK, api.GenerateResponse{
            Model:      req.Model,
            CreatedAt:  time.Now().UTC(),
            Response:   "",
            Done:       true,
            DoneReason: "unload",
        })
        return
    }

    // 모델 능력 검증
    caps := []model.Capability{model.CapabilityCompletion}
    if req.Suffix != "" {
        caps = append(caps, model.CapabilityInsert)
    }

    r, m, opts, err := s.scheduleRunner(c.Request.Context(), name.String(), caps, req.Options, req.KeepAlive)
    if errors.Is(err, errCapabilityCompletion) {
        c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("%q does not support generate", req.Model)})
        return
    } else if err != nil {
        handleScheduleError(c, req.Model, err)
        return
    }

    checkpointLoaded := time.Now()
    
    // 프롬프트가 비어있으면 모델만 로드
    if req.Prompt == "" {
        c.JSON(http.StatusOK, api.GenerateResponse{
            Model:      req.Model,
            CreatedAt:  time.Now().UTC(),
            Done:       true,
            DoneReason: "load",
        })
        return
    }
```

**핵심 특징:**
- **성능 측정**: 체크포인트 기반 성능 추적
- **점진적 검증**: 단계별 입력 검증과 에러 처리
- **능력 기반 라우팅**: 모델 능력에 따른 요청 처리
- **지연 로딩**: 필요시에만 모델 로드

### 2.3 스케줄러 통합

```go
func (s *Server) scheduleRunner(ctx context.Context, name string, caps []model.Capability, requestOpts map[string]any, keepAlive *api.Duration) (llm.LlamaServer, *Model, *api.Options, error) {
    if name == "" {
        return nil, nil, nil, fmt.Errorf("model %w", errRequired)
    }

    model, err := GetModel(name)
    if err != nil {
        return nil, nil, nil, err
    }

    if err := model.CheckCapabilities(caps...); err != nil {
        return nil, nil, nil, fmt.Errorf("%s %w", name, err)
    }

    opts, err := modelOptions(model, requestOpts)
    if err != nil {
        return nil, nil, nil, err
    }

    runnerCh, errCh := s.sched.GetRunner(ctx, model, opts, keepAlive)
    var runner *runnerRef
    select {
    case runner = <-runnerCh:
    case err = <-errCh:
        return nil, nil, nil, err
    }

    return runner.llama, model, &opts, nil
}
```

**스케줄링 전략:**
- **비동기 스케줄링**: 채널 기반 비동기 모델 할당
- **능력 검증**: 요청된 기능을 모델이 지원하는지 사전 검증
- **옵션 통합**: 모델 기본 옵션과 요청 옵션 병합
- **컨텍스트 인식**: 취소 가능한 요청 처리

## 3. 이미지 처리 및 멀티모달 지원

### 3.1 MLLama 모델 특화 처리

```go
// MLLama 모델 확인
isMllama := checkMllamaModelFamily(m)
if isMllama && len(req.Images) > 1 {
    c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "this model only supports one image: more than one image sent"})
    return
}

images := make([]llm.ImageData, len(req.Images))
for i := range req.Images {
    if isMllama && len(m.ProjectorPaths) > 0 {
        // MLLama 전용 이미지 전처리
        data, opts, err := mllama.Preprocess(bytes.NewReader(req.Images[i]))
        if err != nil {
            c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": "error processing image"})
            return
        }

        ar, ok := opts["aspectRatioIndex"].(int)
        if !ok {
            c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": "error processing image"})
            return
        }

        buf := new(bytes.Buffer)
        err = binary.Write(buf, binary.LittleEndian, data)
        if err != nil {
            c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": "error processing image"})
            return
        }

        images[i] = llm.ImageData{ID: i, Data: buf.Bytes(), AspectRatioID: ar}
    } else {
        images[i] = llm.ImageData{ID: i, Data: req.Images[i]}
    }
}
```

**멀티모달 특징:**
- **모델별 최적화**: MLLama 모델을 위한 특화된 이미지 처리
- **종횡비 인덱싱**: 이미지 종횡비 정보 보존
- **바이너리 처리**: Little Endian 바이너리 변환
- **타입 안전성**: 런타임 타입 체크와 에러 처리

### 3.2 템플릿 처리 시스템

```go
if !req.Raw {
    tmpl := m.Template
    if req.Template != "" {
        tmpl, err = template.Parse(req.Template)
        if err != nil {
            c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
            return
        }
    }

    var values template.Values
    if req.Suffix != "" {
        values.Prompt = prompt
        values.Suffix = req.Suffix
    } else {
        var msgs []api.Message
        if req.System != "" {
            msgs = append(msgs, api.Message{Role: "system", Content: req.System})
        } else if m.System != "" {
            msgs = append(msgs, api.Message{Role: "system", Content: m.System})
        }

        if req.Context == nil {
            msgs = append(msgs, m.Messages...)
        }

        // 이미지 프롬프트 추가
        for _, i := range images {
            imgPrompt := ""
            if isMllama {
                imgPrompt = "<|image|>"
            }
            msgs = append(msgs, api.Message{Role: "user", Content: fmt.Sprintf("[img-%d]"+imgPrompt, i.ID)})
        }

        values.Messages = append(msgs, api.Message{Role: "user", Content: req.Prompt})
    }

    var b bytes.Buffer
    if err := tmpl.Execute(&b, values); err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
        return
    }

    prompt = b.String()
}
```

**템플릿 시스템:**
- **동적 템플릿**: 런타임 템플릿 파싱과 적용
- **메시지 구조화**: 시스템/사용자 메시지 자동 구성
- **이미지 통합**: 텍스트와 이미지의 자연스러운 통합
- **컨텍스트 관리**: 대화 히스토리와 새 메시지 병합

## 4. 스트리밍 응답 처리

### 4.1 비동기 스트리밍 아키텍처

```go
ch := make(chan any)
go func() {
    var sb strings.Builder
    defer close(ch)
    if err := r.Completion(c.Request.Context(), llm.CompletionRequest{
        Prompt:  prompt,
        Images:  images,
        Format:  req.Format,
        Options: opts,
    }, func(cr llm.CompletionResponse) {
        res := api.GenerateResponse{
            Model:     req.Model,
            CreatedAt: time.Now().UTC(),
            Response:  cr.Content,
            Done:      cr.Done,
            Metrics: api.Metrics{
                PromptEvalCount:    cr.PromptEvalCount,
                PromptEvalDuration: cr.PromptEvalDuration,
                EvalCount:          cr.EvalCount,
                EvalDuration:       cr.EvalDuration,
            },
        }

        if _, err := sb.WriteString(cr.Content); err != nil {
            ch <- gin.H{"error": err.Error()}
        }

        if cr.Done {
            res.DoneReason = cr.DoneReason.String()
            res.TotalDuration = time.Since(checkpointStart)
            res.LoadDuration = checkpointLoaded.Sub(checkpointStart)

            if !req.Raw {
                tokens, err := r.Tokenize(c.Request.Context(), prompt+sb.String())
                if err != nil {
                    ch <- gin.H{"error": err.Error()}
                    return
                }
                res.Context = tokens
            }
        }

        ch <- res
    }); err != nil {
        ch <- gin.H{"error": err.Error()}
    }
}()
```

**스트리밍 특징:**
- **고루틴 분리**: 메인 요청과 분리된 스트리밍 처리
- **콜백 패턴**: 실시간 응답 생성을 위한 콜백 메커니즘
- **성능 메트릭**: 토큰 처리 속도와 지연 시간 측정
- **에러 전파**: 스트리밍 중 에러의 안전한 전달

### 4.2 응답 형태 처리

```go
func streamResponse(c *gin.Context, ch chan any) {
    c.Header("Content-Type", "application/x-ndjson")
    c.Stream(func(w io.Writer) bool {
        val, ok := <-ch
        if !ok {
            return false
        }

        bts, err := json.Marshal(val)
        if err != nil {
            slog.Info(fmt.Sprintf("streamResponse: json.Marshal failed with %s", err))
            return false
        }

        // 개행 문자로 청크 구분
        bts = append(bts, '\n')
        if _, err := w.Write(bts); err != nil {
            slog.Info(fmt.Sprintf("streamResponse: w.Write failed with %s", err))
            return false
        }

        return true
    })
}

func waitForStream(c *gin.Context, ch chan any) {
    c.Header("Content-Type", "application/json")
    var latest api.ProgressResponse
    for resp := range ch {
        switch r := resp.(type) {
        case api.ProgressResponse:
            latest = r
        case gin.H:
            status, ok := r["status"].(int)
            if !ok {
                status = http.StatusInternalServerError
            }
            errorMsg, ok := r["error"].(string)
            if !ok {
                errorMsg = "unknown error"
            }
            c.JSON(status, gin.H{"error": errorMsg})
            return
        default:
            c.JSON(http.StatusInternalServerError, gin.H{"error": "unknown message type"})
            return
        }
    }

    c.JSON(http.StatusOK, latest)
}
```

**응답 처리 전략:**
- **NDJSON 스트리밍**: 개행 구분자 기반 JSON 스트리밍
- **조건부 스트리밍**: 클라이언트 요청에 따른 스트림/배치 선택
- **타입 안전성**: 런타임 타입 검증과 에러 처리
- **상태 추적**: 마지막 응답 상태 보존

## 5. 임베딩 및 벡터 처리

### 5.1 임베딩 생성 핸들러

```go
func (s *Server) EmbedHandler(c *gin.Context) {
    checkpointStart := time.Now()
    var req api.EmbedRequest
    // ... 입력 검증 코드 ...

    // 입력 타입 처리
    var input []string
    switch i := req.Input.(type) {
    case string:
        if len(i) > 0 {
            input = append(input, i)
        }
    case []any:
        for _, v := range i {
            if _, ok := v.(string); !ok {
                c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "invalid input type"})
                return
            }
            input = append(input, v.(string))
        }
    default:
        if req.Input != nil {
            c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "invalid input type"})
            return
        }
    }

    // 병렬 임베딩 생성
    var g errgroup.Group
    embeddings := make([][]float32, len(input))
    for i, text := range input {
        g.Go(func() error {
            embedding, err := r.Embedding(c.Request.Context(), text)
            if err != nil {
                return err
            }
            embeddings[i] = normalize(embedding)  // 벡터 정규화
            return nil
        })
    }

    if err := g.Wait(); err != nil {
        c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": strings.TrimSpace(err.Error())})
        return
    }

    resp := api.EmbedResponse{
        Model:           req.Model,
        Embeddings:      embeddings,
        TotalDuration:   time.Since(checkpointStart),
        LoadDuration:    checkpointLoaded.Sub(checkpointStart),
        PromptEvalCount: count,
    }
    c.JSON(http.StatusOK, resp)
}

// 벡터 정규화 함수
func normalize(vec []float32) []float32 {
    var sum float32
    for _, v := range vec {
        sum += v * v
    }

    norm := float32(0.0)
    if sum > 0 {
        norm = float32(1.0 / math.Sqrt(float64(sum)))
    }

    for i := range vec {
        vec[i] *= norm
    }
    return vec
}
```

**임베딩 처리 특징:**
- **동적 입력 타입**: 문자열과 배열 입력 모두 지원
- **병렬 처리**: `errgroup`을 활용한 동시 임베딩 생성
- **벡터 정규화**: L2 노름 기반 벡터 정규화
- **성능 측정**: 처리 시간과 토큰 수 추적

### 5.2 컨텍스트 길이 관리

```go
kvData, _, err := getModelData(m.ModelPath, false)
if err != nil {
    c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
    return
}

var count int
for i, s := range input {
    tokens, err := r.Tokenize(c.Request.Context(), s)
    if err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
        return
    }

    ctxLen := min(opts.NumCtx, int(kvData.ContextLength()))
    if len(tokens) > ctxLen {
        if !truncate {
            c.JSON(http.StatusBadRequest, gin.H{"error": "input length exceeds maximum context length"})
            return
        }

        tokens = tokens[:ctxLen]
        s, err = r.Detokenize(c.Request.Context(), tokens)
        if err != nil {
            c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
            return
        }
    }

    count += len(tokens)
    input[i] = s
}
```

**컨텍스트 관리:**
- **동적 제한**: 모델별 컨텍스트 길이 동적 적용
- **자동 절단**: 옵션에 따른 입력 텍스트 자동 절단
- **토큰 재구성**: 절단된 토큰의 텍스트 재변환
- **사용량 추적**: 전체 토큰 사용량 계산

## 6. 보안 및 접근 제어

### 6.1 호스트 검증 미들웨어

```go
func allowedHostsMiddleware(addr net.Addr) gin.HandlerFunc {
    return func(c *gin.Context) {
        if addr == nil {
            c.Next()
            return
        }

        // 루프백이 아닌 주소에서 바인딩된 경우 통과
        if addr, err := netip.ParseAddrPort(addr.String()); err == nil && !addr.Addr().IsLoopback() {
            c.Next()
            return
        }

        host, _, err := net.SplitHostPort(c.Request.Host)
        if err != nil {
            host = c.Request.Host
        }

        // IP 주소 검증
        if addr, err := netip.ParseAddr(host); err == nil {
            if addr.IsLoopback() || addr.IsPrivate() || addr.IsUnspecified() || isLocalIP(addr) {
                c.Next()
                return
            }
        }

        // 허용된 호스트명 검증
        if allowedHost(host) {
            if c.Request.Method == http.MethodOptions {
                c.AbortWithStatus(http.StatusNoContent)
                return
            }

            c.Next()
            return
        }

        c.AbortWithStatus(http.StatusForbidden)
    }
}

func allowedHost(host string) bool {
    host = strings.ToLower(host)

    if host == "" || host == "localhost" {
        return true
    }

    if hostname, err := os.Hostname(); err == nil && host == strings.ToLower(hostname) {
        return true
    }

    tlds := []string{"localhost", "local", "internal"}
    for _, tld := range tlds {
        if strings.HasSuffix(host, "."+tld) {
            return true
        }
    }

    return false
}
```

**보안 검증:**
- **네트워크 기반 제어**: IP 주소 유형별 접근 제어
- **호스트명 화이트리스트**: 안전한 호스트명만 허용
- **CORS preflight 처리**: OPTIONS 요청 적절한 처리
- **로컬 환경 감지**: 개발 환경에서의 유연한 접근

### 6.2 실험적 기능 제어

```go
func experimentEnabled(name string) bool {
    return slices.Contains(strings.Split(os.Getenv("OLLAMA_EXPERIMENT"), ","), name)
}

var useClient2 = experimentEnabled("client2")
```

**기능 토글:**
- **환경 변수 기반**: `OLLAMA_EXPERIMENT`로 실험 기능 활성화
- **쉼표 구분**: 여러 실험 기능 동시 활성화
- **런타임 제어**: 재컴파일 없는 기능 토글

## 7. 에러 처리 및 복구 전략

### 7.1 계층화된 에러 처리

```go
func handleScheduleError(c *gin.Context, name string, err error) {
    switch {
    case errors.Is(err, errCapabilities), errors.Is(err, errRequired):
        c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
    case errors.Is(err, context.Canceled):
        c.JSON(499, gin.H{"error": "request canceled"})  // 비표준 상태 코드
    case errors.Is(err, ErrMaxQueue):
        c.JSON(http.StatusServiceUnavailable, gin.H{"error": err.Error()})
    case errors.Is(err, os.ErrNotExist):
        c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model %q not found, try pulling it first", name)})
    default:
        c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
    }
}
```

**에러 분류 전략:**
- **에러 타입 기반**: `errors.Is`를 활용한 정확한 에러 분류
- **적절한 HTTP 코드**: 에러 유형에 맞는 상태 코드 반환
- **사용자 친화적 메시지**: 기술적 에러의 이해하기 쉬운 변환
- **특수 상태 코드**: 499(Client Closed Request) 같은 특수 케이스 처리

### 7.2 모델 상태 관리

```go
func (s *Server) PsHandler(c *gin.Context) {
    models := []api.ProcessModelResponse{}

    for _, v := range s.sched.loaded {
        model := v.model
        mr := api.ProcessModelResponse{
            Model:     model.ShortName,
            Name:      model.ShortName,
            Size:      int64(v.estimatedTotal),
            SizeVRAM:  int64(v.estimatedVRAM),
            Digest:    model.Digest,
            ExpiresAt: v.expiresAt,
        }
        
        // 로딩 중인 모델 처리
        var epoch time.Time
        if v.expiresAt == epoch {
            mr.ExpiresAt = time.Now().Add(v.sessionDuration)
        }

        models = append(models, mr)
    }

    // 만료 시간 기준 정렬 (오래 남은 것부터)
    slices.SortStableFunc(models, func(i, j api.ProcessModelResponse) int {
        return cmp.Compare(j.ExpiresAt.Unix(), i.ExpiresAt.Unix())
    })

    c.JSON(http.StatusOK, api.ProcessResponse{Models: models})
}
```

**상태 모니터링:**
- **메모리 사용량**: 총 메모리와 VRAM 사용량 추적
- **만료 시간**: 모델 언로드 예정 시간 관리
- **로딩 상태**: 로딩 중인 모델의 적절한 시간 계산
- **정렬된 표시**: 사용자 편의를 위한 정렬된 모델 목록

## 8. 결론

Ollama Custom의 서버 아키텍처는 현대적인 Go 웹 개발 패턴을 충실히 구현한 고품질 API 서버입니다:

### 8.1 주요 강점

- **확장 가능한 구조**: 모듈화된 핸들러와 미들웨어 시스템
- **성능 최적화**: 비동기 처리와 스트리밍 응답
- **호환성**: OpenAI API와의 완벽한 호환성
- **보안**: 다층 보안 검증과 접근 제어

### 8.2 설계 철학

- **관심사 분리**: 라우팅, 비즈니스 로직, 스케줄링의 명확한 분리
- **에러 투명성**: 계층화된 에러 처리로 디버깅 용이성 제공
- **성능 중심**: 메트릭 수집과 성능 최적화 내장
- **확장성**: 새로운 엔드포인트와 기능 추가 용이

다음 포스트에서는 모델 관리 시스템과 생명주기 관리를 분석해보겠습니다.