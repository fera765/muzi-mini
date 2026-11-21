package com.muzi.mini.engine

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import org.json.JSONObject
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * Motor CPU-only sobre ONNX Runtime core (NEON) com loop autoregressivo e
 * KV cache. Assinatura do modelo (Qwen2.5 ONNX / transformers.js):
 *   in : input_ids, attention_mask, position_ids, past_key_values.{L}.{key,value}
 *   out: logits, present.{L}.{key,value}
 * KV em fp32 (use model_quantized.onnx). Decode guloso (argmax).
 */
class OnnxLlmEngine(private val threads: Int) : AutoCloseable {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null
    private var tokenizer: QwenTokenizer? = null

    private var nLayers = 24
    private var nKvHeads = 2
    private var headDim = 64

    fun load(modelDir: String) {
        close()
        val dir = File(modelDir)
        val onnx = dir.listFiles { f -> f.name.endsWith(".onnx") }?.firstOrNull()
            ?: error("Nenhum .onnx em $modelDir")
        File(dir, "config.json").takeIf { it.exists() }?.let { readConfig(it) }
        File(dir, "tokenizer.json").takeIf { it.exists() }?.let { tokenizer = QwenTokenizer(it) }
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(threads)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        session = env.createSession(onnx.absolutePath, opts)
    }

    private fun readConfig(f: File) {
        val c = JSONObject(f.readText())
        nLayers = c.optInt("num_hidden_layers", nLayers)
        nKvHeads = c.optInt("num_key_value_heads", nKvHeads)
        val hidden = c.optInt("hidden_size", 896)
        val heads = c.optInt("num_attention_heads", 14)
        headDim = c.optInt("head_dim", if (heads > 0) hidden / heads else 64)
    }

    val isLoaded get() = session != null && tokenizer != null

    private fun longTensor(data: LongArray, shape: LongArray) =
        OnnxTensor.createTensor(env, LongBuffer.wrap(data), shape)

    private fun emptyKV() =
        OnnxTensor.createTensor(env, FloatBuffer.allocate(0),
            longArrayOf(1, nKvHeads.toLong(), 0, headDim.toLong()))

    fun generate(prompt: String, maxTokens: Int = 256): Flow<String> = flow {
        val s = session ?: error("Modelo nao carregado")
        val tok = tokenizer ?: error("Tokenizer nao carregado")

        var past = Array(nLayers) { Pair(emptyKV(), emptyKV()) }
        var curIds = tok.encodeChat(prompt)
        var pastLen = 0L

        try {
            for (step in 0 until maxTokens) {
                val curLen = curIds.size
                val feeds = HashMap<String, OnnxTensor>()
                feeds["input_ids"] = longTensor(curIds, longArrayOf(1, curLen.toLong()))
                val attnLen = pastLen + curLen
                feeds["attention_mask"] = longTensor(LongArray(attnLen.toInt()) { 1L }, longArrayOf(1, attnLen))
                feeds["position_ids"] = longTensor(LongArray(curLen) { pastLen + it }, longArrayOf(1, curLen.toLong()))
                for (l in 0 until nLayers) {
                    feeds["past_key_values.$l.key"] = past[l].first
                    feeds["past_key_values.$l.value"] = past[l].second
                }

                val result = s.run(feeds)
                val logits = result.get("logits").get() as OnnxTensor
                val next = argmaxLastRow(logits)

                // novos past = present.* (copiados para sobreviver ao close)
                val newPast = Array(nLayers) { l ->
                    Pair(
                        copyFloatTensor(result.get("present.$l.key").get() as OnnxTensor),
                        copyFloatTensor(result.get("present.$l.value").get() as OnnxTensor)
                    )
                }
                result.close()
                // feeds inclui os tensores de past desta iteracao -> fecha todos aqui
                feeds.values.forEach { it.close() }
                past = newPast
                pastLen += curLen

                if (next == tok.imEnd || next == tok.endOfText) break
                val piece = tok.decodeToken(next)
                if (piece.isNotEmpty()) emit(piece)
                curIds = longArrayOf(next.toLong())
            }
        } finally {
            past.forEach { it.first.close(); it.second.close() }
        }
    }.flowOn(Dispatchers.Default)

    /** Copia um OnnxTensor float para um novo (desacopla do result que sera fechado). */
    private fun copyFloatTensor(t: OnnxTensor): OnnxTensor {
        val shape = t.info.shape
        val buf = t.floatBuffer
        val arr = FloatArray(buf.remaining())
        buf.get(arr)
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(arr), shape)
    }

    /** argmax da ultima posicao de logits [1, seq, vocab]. */
    private fun argmaxLastRow(logits: OnnxTensor): Int {
        val shape = logits.info.shape           // [1, seq, vocab]
        val seq = shape[1].toInt(); val vocab = shape[2].toInt()
        val buf = logits.floatBuffer
        val offset = (seq - 1) * vocab
        var best = -1; var bestVal = Float.NEGATIVE_INFINITY
        for (i in 0 until vocab) {
            val v = buf.get(offset + i)
            if (v > bestVal) { bestVal = v; best = i }
        }
        return best
    }

    override fun close() {
        session?.close(); session = null
        tokenizer = null
    }
}
