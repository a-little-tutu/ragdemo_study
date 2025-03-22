package main

import (
	"context"
	"github.com/tmc/langchaingo/documentloaders"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/textsplitter"
	"llms/ollama"
	"os"
)

const ak = "WJXH4FM9MGP4QJJAYJ3J"
const sk = "lIy83m4QDUU8KQ21zwpkXOGjasjLWfIt4Xrxltu0"

//func main() {
//	// endpoint填写Bucket对应的Endpoint, 这里以华北-北京四为例，其他地区请按实际情况填写。
//	endPoint := "https://obs.cn-north-4.myhuaweicloud.com"
//	// 创建obsClient实例
//	// 如果使用临时AKSK和SecurityToken访问OBS，需要在创建实例时通过obs.WithSecurityToken方法指定securityToken值。
//	obsClient, err := obs.New(ak, sk, endPoint, obs.WithSignature(obs.SignatureObs) /*, obs.WithSecurityToken(securityToken)*/)
//	if err != nil {
//		panic(err)
//	}
//
//	// 指定存储桶名称
//	bucketname := "cbdrbucket"
//	// 判断桶是否存在
//	output, err := obsClient.HeadBucket(bucketname)
//	if err == nil {
//		fmt.Printf("Head bucket(%s) successful!\n", bucketname)
//		fmt.Printf("RequestId:%s\n", output.RequestId)
//		return
//	}
//	fmt.Printf("Head bucket(%s) fail!\n", bucketname)
//	if obsError, ok := err.(obs.ObsError); ok {
//		fmt.Println("An ObsError was found, which means your request sent to OBS was rejected with an error response.")
//		fmt.Println(obsError.Error())
//	} else {
//		fmt.Println("An Exception was found, which means the client encountered an internal problem when attempting to communicate with OBS, for example, the client was unable to access the network.")
//		fmt.Println(err)
//	}
//
//	obsClient.Close()
//}

//ragdemo 项目是一个基于 Go 语言和 langchaingo 库实现的检索增强生成（RAG, Retrieval Augmented Generation）示例项目。
//该项目的主要目的是利用大语言模型（LLM）和向量数据库（Qdrant），实现对文档的存储、检索，并根据检索结果生成相关的回答。

const prompt = `请介绍一下这篇文档`

func main() {
	//将指定文件内容读取并分割成多个文档块。
	docs, _ := TextToChunks("./quantstudy.pdf", 1000, 500)
	//将文档块存储到向量存储中。
	err := ollama.StoreDocs(docs, ollama.GetStore())
	if err != nil {
		panic(err)
	}
	//从向量存储中检索与给定提示（prompt）相关的文档。
	rst, err := ollama.UseRetriaver(ollama.GetStore(), prompt, 5)
	if err != nil {
		panic(err)
	}

	//// 调用 ollama 包中的 GetAnswer 函数，使用检索到的文档和提示信息，让大语言模型生成回答
	//// context.Background() 创建一个空的上下文，用于在函数调用过程中传递请求范围的数据和取消信号
	//// ollama.GetOllamaMistral() 返回一个 Ollama 的 Mistral 模型实例
	//// rst 是前面检索到的文档列表
	//// prompt 是预先定义的提示信息
	answer, err := ollama.GetAnswer(context.Background(), ollama.GetOllamaMistral(), rst, prompt)
	if err != nil {
		panic(err)
	}
	println(answer)
}

// TextToChunks 函数用于将指定文件的内容读取并分割成多个文档块。
// 参数:
//   - dirFile: 要处理的文件的路径。
//   - chunkSize: 每个文档块的最大大小。
//   - chunkOverlap: 相邻文档块之间的重叠部分大小。
//
// 返回值:
//   - []schema.Document: 分割后的文档块切片。
//   - error: 处理过程中可能出现的错误。
func TextToChunks(dirFile string, chunkSize, chunkOverlap int) ([]schema.Document, error) {
	// 打开指定路径的文件
	file, err := os.Open(dirFile)
	if err != nil {
		return nil, err
	}
	// 创建一个文本加载器，用于加载文件内容
	docLoaded := documentloaders.NewText(file)
	// 创建一个递归字符分割器，用于将文本分割成块
	split := textsplitter.NewRecursiveCharacter()
	// 设置分割器的每个文档块的最大大小
	split.ChunkSize = chunkSize
	// 设置分割器的相邻文档块之间的重叠部分大小
	split.ChunkOverlap = chunkOverlap
	// 调用加载器的 LoadAndSplit 方法，使用分割器将文件内容分割成文档块
	docs, err := docLoaded.LoadAndSplit(context.Background(), split)
	if err != nil {
		return nil, err
	}
	return docs, nil
}
