package com.muzi.mini.engine

import ai.onnxruntime.genai.Generator
import ai.onnxruntime.genai.GeneratorParams
import ai.onnxruntime.genai.Model
import ai.onnxruntime.genai.Tokenizer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn

/**
 * Motor de inferencia CPU-only sobre ONNX Runtime GenAI.
 * Carrega a pasta do modelo (ONNX + tokenizer) e gera tokens em streaming.
 *
 * Observacao de versao: a API do onnxruntime-genai muda entre releases.
 * Este wrapper segue a 0.4.x; ajuste os nomes de metodo se fixar outra versao.
 */
class OnnxLlmEngine(private val threads: Int) : AutoCloseable {
    private var model: Model? = null
    private var tokenizer: Tokenizer? = null

    fun load(modelDir: String) {
        close()
        model = Model(modelDir)
        tokenizer = Tokenizer(model!!)
    }

    val isLoaded get() = model != null

    /** Gera a resposta token a token. maxTokens limita a saida. */
    fun generate(prompt: String, maxTokens: Int = 1024): Flow<String> = flow {
        val m = model ?: error("Modelo nao carregado")
        val tok = tokenizer!!
        val params = GeneratorParams(m).apply {
            setSearchOption("max_length", (maxTokens).toDouble())
            setSearchOption("do_sample", 1.0)
            setSearchOption("temperature", 0.7)
        }
        val input = tok.encode(prompt)
        val gen = Generator(m, params)
        gen.appendTokenSequences(input)
        val stream = tok.createStream()
        try {
            while (!gen.isDone) {
                gen.generateNextToken()
                val last = gen.getLastTokenInSequence(0)
                val piece = stream.decode(last)
                if (piece.isNotEmpty()) emit(piece)
            }
        } finally {
            stream.close(); gen.close(); params.close()
        }
    }.flowOn(Dispatchers.Default)

    override fun close() {
        tokenizer?.close(); tokenizer = null
        model?.close(); model = null
    }
}
