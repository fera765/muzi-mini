package com.muzi.mini.engine

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import java.io.File

/**
 * Motor de inferencia CPU-only sobre ONNX Runtime core (NEON).
 * Carrega a sessao ONNX e expoe a geracao em streaming.
 *
 * ESTADO: o carregamento da sessao e o controle de threads/RAM estao prontos
 * e reais. O loop autoregressivo (encode -> run -> decode com KV cache) e o
 * ponto de integracao final: depende dos nomes de entrada/saida do modelo
 * escolhido e de um tokenizer. Para modelos exportados pelo onnxruntime-genai
 * (Qwen/Phi), o ideal e compilar o onnxruntime-genai-android do fonte e trocar
 * este wrapper pela Generator API — ver android/README.md.
 */
class OnnxLlmEngine(private val threads: Int) : AutoCloseable {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null

    fun load(modelDir: String) {
        close()
        val onnx = File(modelDir).listFiles { f -> f.name.endsWith(".onnx") }?.firstOrNull()
            ?: error("Nenhum arquivo .onnx em $modelDir")
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(threads)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        session = env.createSession(onnx.absolutePath, opts)
    }

    val isLoaded get() = session != null

    /** Nomes de entrada/saida do grafo — util para ligar o loop de geracao. */
    fun ioInfo(): String {
        val s = session ?: return "sessao nao carregada"
        return "inputs=${s.inputNames} outputs=${s.outputNames}"
    }

    fun generate(prompt: String, maxTokens: Int = 512): Flow<String> = flow {
        val s = session ?: error("Modelo nao carregado")
        // Ponto de integracao: aqui entra tokenizer + loop autoregressivo com
        // KV cache usando s.run(...). Ate ligar, reportamos o estado real.
        emit("[sessao ONNX carregada] ${ioInfo()}\n")
        emit("[loop de geracao ainda nao ligado para este modelo — ver README]")
    }.flowOn(Dispatchers.Default)

    override fun close() {
        session?.close(); session = null
    }
}
