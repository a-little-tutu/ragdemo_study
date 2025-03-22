package ollama

import (
	"context"
	"fmt"
	"github.com/tmc/langchaingo/agents"
	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/ollama"
	"github.com/tmc/langchaingo/memory"
	"net/url"

	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/vectorstores"
	"github.com/tmc/langchaingo/vectorstores/qdrant"
)

var (
	collectionName = "langchaingo-ollama-rag"
	// 定义了一个字符串类型的变量 qdrantUrl，用于指定 Qdrant 向量数据库服务的访问地址。
	// 这里使用了本地的 Qdrant 服务，端口号为 6333。
	qdrantUrl    = "http://localhost:6333"
	ollamaServer = "http://localhost:11434"
)

// getOllamaEmbedder 函数用于创建并返回一个 embeddings.EmbedderImpl 实例，
// 该实例使用 Ollama 大语言模型进行文本嵌入（将文本转换成向量）。
func getOllamaEmbedder() *embeddings.EmbedderImpl {
	// 创建一个新的 Ollama 大语言模型实例。
	// 使用的模型是 "nomic-embed-text:latest"，并连接到由 ollamaServer 指定的 Ollama 服务器。
	ollamaEmbedderModel, err := ollama.New(ollama.WithModel("nomic-embed-text:latest"), ollama.WithServerURL(ollamaServer))
	if err != nil {
		panic(err)
	}
	// 使用 Ollama 模型创建一个新的嵌入器实例。
	ollamaEmbedder, err := embeddings.NewEmbedder(ollamaEmbedderModel)
	if err != nil {
		panic(err)
	}
	// 返回创建好的嵌入器实例和 nil 错误信息。
	return ollamaEmbedder
}

// 大模型设置为deepseek大模型示例
func GetOllamaMistral() *ollama.LLM {
	llm, err := ollama.New(
		ollama.WithModel("deepseek-r1:1.5b"),
		ollama.WithServerURL(ollamaServer))
	if err != nil {
		panic(err)
	}
	return llm
}

func GetOllamaLlama2() *ollama.LLM {
	// 创建一个新的ollama模型，模型名为"llama2-chinese:13b"
	llm, err := ollama.New(
		ollama.WithModel("llama2-chinese:13b"),
		ollama.WithServerURL(ollamaServer))
	if err != nil {
		panic(err)
	}
	return llm
}

// GetStore 函数用于创建并返回一个 Qdrant 向量存储实例。
// 该实例会使用指定的 URL、API 密钥、集合名称和嵌入器进行初始化。
// 返回值:
//   - *qdrant.Store: 指向 Qdrant 向量存储实例的指针。
func GetStore() *qdrant.Store {
	// 解析 qdrantUrl 字符串为 url.URL 类型
	// qdrantUrl 是一个全局变量，指定了 Qdrant 服务的地址
	qdUrl, err := url.Parse(qdrantUrl)
	if err != nil {
		panic(err)
	}
	// 创建一个新的 Qdrant 向量存储实例
	// 使用 qdrant 包提供的 New 函数，并传入相关配置选项
	store, err := qdrant.New(
		// 设置 Qdrant 服务的 URL
		qdrant.WithURL(*qdUrl),
		// 设置 Qdrant 的 API 密钥，这里为空字符串表示不使用 API 密钥
		qdrant.WithAPIKey(""),
		// 设置存储文档的集合名称，collectionName 是一个全局变量
		qdrant.WithCollectionName(collectionName),
		// 设置用于文档嵌入的嵌入器，调用 getOllamaEmbedder 函数获取
		qdrant.WithEmbedder(getOllamaEmbedder()),
	)
	if err != nil {
		panic(err)
	}
	return &store
}

// StoreDocs 函数用于将文档切片存储到 Qdrant 向量存储中。
// 参数:
//   - docs: 要存储的文档切片，每个元素为一个 schema.Document 类型的文档。
//   - store: 指向 Qdrant 向量存储实例的指针，用于实际存储文档。
//
// 返回值:
//   - error: 如果存储过程中出现错误，返回相应的错误信息；否则返回 nil。
func StoreDocs(docs []schema.Document, store *qdrant.Store) error {
	if len(docs) > 0 {
		// 调用 store 的 AddDocuments 方法将文档切片转换为向量并添加到向量存储中
		// context.Background() 创建一个空的上下文，用于在操作过程中传递请求范围的数据和取消信号
		_, err := store.AddDocuments(context.Background(), docs)
		if err != nil {
			return err
		}
	}
	return nil
}

// UseRetriaver 函数用于从 Qdrant 向量存储中检索与给定提示相关的文档。
// 参数:
//   - store: 指向 Qdrant 向量存储实例的指针，用于存储和检索文档向量。
//   - prompt: 用于检索的提示信息，通常是一个关键词或问题。
//   - topk: 指定要返回的相关文档（相关chunk）的最大数量。
//
// 返回值:
//   - []schema.Document: 包含检索到的相关文档的切片。
//   - error: 如果检索过程中出现错误，返回相应的错误信息；否则返回 nil。
func UseRetriaver(store *qdrant.Store, prompt string, topk int) ([]schema.Document, error) {
	// 配置检索选项，设置分数阈值为 0.80，只有相似度得分大于等于 0.80 的文档才会被考虑。
	optionsVector := []vectorstores.Option{
		vectorstores.WithScoreThreshold(0.80),
	}
	// 将 Qdrant 向量存储实例转换为检索器，指定返回的最大文档数量和检索选项。
	retriever := vectorstores.ToRetriever(store, topk, optionsVector...)
	// 执行文档检索操作，使用空的上下文对象和给定的提示信息。
	docRetrieved, err := retriever.GetRelevantDocuments(context.Background(), prompt)

	if err != nil {
		return nil, fmt.Errorf("检索文档失败: %v", err)
	}
	// 如果检索成功，返回检索到的相关文档和 nil 错误信息
	return docRetrieved, nil
}

// GetAnswer 函数用于根据给定的上下文、大语言模型、检索到的文档和提示信息，获取对应的答案。
// 参数:
//   - ctx: 上下文对象，用于控制操作的生命周期，可传递取消信号、截止时间等信息。
//   - llm: 大语言模型实例，用于生成答案。
//   - docRetrieved: 检索到的相关文档切片，包含与提示相关的文档信息。
//   - prompt: 用户提供的提示信息，即希望模型回答的问题。
//
// 返回值:
//   - string: 模型生成的答案。
//   - error: 如果执行过程中出现错误，返回相应的错误信息；否则返回 nil。
func GetAnswer(ctx context.Context, llm llms.Model, docRetrieved []schema.Document, prompt string) (string, error) {
	history := memory.NewChatMessageHistory() // 创建一个新的聊天消息历史记录实例

	// 遍历检索到的文档，将每个文档的内容作为 AI 消息添加到聊天历史记录中
	//目的是将相关文档信息作为历史对话的一部分，让模型在生成答案时能够参考这些信息。
	for _, doc := range docRetrieved {
		history.AddAIMessage(ctx, doc.PageContent)
	}

	// 创建一个新的对话缓冲区，使用之前创建的聊天历史记录
	conversation := memory.NewConversationBuffer(memory.WithChatHistory(history))

	// 创建一个新的执行器，使用对话式代理和自定义的操作选项
	//执行器的主要功能是协调和执行对话式代理（agents.NewConversationalAgent）的操作。
	//它会结合对话的上下文信息（存储在对话缓冲区中），根据用户的提示（prompt），指挥大语言模型（llm）生成合适的回答。
	executor := agents.NewExecutor(
		// 创建一个新的对话式代理，使用给定的大语言模型
		//对话式代理会将用户提示和对话历史进行整合，将其转换为适合大语言模型处理的格式。
		//对话式代理会将处理后的输入传递给大语言模型，将大语言模型生成的回复返回给调用者，
		agents.NewConversationalAgent(llm, nil),
		// 调用 NoOpOption 函数，使用默认的无操作选项
		NoOpOption(),
		// 将对话缓冲区作为执行器的内存
		agents.WithMemory(conversation),
	)

	// 定义链式调用的选项，设置温度参数为 0.8，影响模型生成结果的随机性
	options := []chains.ChainCallOption{
		chains.WithTemperature(0.8),
	}

	// 执行链式调用，传入上下文、执行器、提示信息和调用选项
	res, err := chains.Run(ctx, executor, prompt, options...)
	if err != nil {
		return "", err
	}

	// 如果执行成功，返回模型生成的答案和 nil 错误信息
	return res, nil
}

// NoOpOption 是一个默认什么都不做的 Option
func NoOpOption() agents.Option {
	return func(co *agents.Options) {
		// 不做任何操作
	}
}

func Translate(llm llms.Model, text string) (string, error) {
	completion, err := llms.GenerateFromSinglePrompt(
		context.TODO(),
		llm,
		"将如下这句话翻译为中文，只需要回复翻译后的内容，而不需要回复其他任何内容。需要翻译的英文内容是: \n"+text,
		llms.WithTemperature(0.8))
	if err != nil {
		return "", err
	}
	return completion, nil
}
