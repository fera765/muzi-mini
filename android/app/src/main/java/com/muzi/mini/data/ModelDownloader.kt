package com.muzi.mini.data

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File

/** Baixa os arquivos de um modelo para o armazenamento do app, com progresso. */
class ModelDownloader(private val baseDir: File) {
    private val client = OkHttpClient()

    fun modelDir(spec: ModelSpec) = File(baseDir, spec.id).apply { mkdirs() }

    fun isInstalled(spec: ModelSpec): Boolean {
        if (spec.files.isEmpty()) return false
        val dir = modelDir(spec)
        return spec.files.all { File(dir, it.name).exists() }
    }

    /** onProgress: fracao 0..1 do total. Retorna a pasta do modelo. */
    suspend fun download(spec: ModelSpec, onProgress: (Float) -> Unit): File =
        withContext(Dispatchers.IO) {
            val dir = modelDir(spec)
            val totalBytes = spec.files.sumOf { it.sizeMB.toLong() } * 1_000_000L
            var done = 0L
            for (f in spec.files) {
                val out = File(dir, f.name)
                if (out.exists() && out.length() > 0) { done += out.length(); onProgress((done.toFloat()/totalBytes)); continue }
                val req = Request.Builder().url(f.url).build()
                client.newCall(req).execute().use { resp ->
                    if (!resp.isSuccessful) throw RuntimeException("Falha ao baixar ${f.name}: ${resp.code}")
                    resp.body!!.byteStream().use { input ->
                        out.outputStream().use { output ->
                            val buf = ByteArray(1 shl 16)
                            while (true) {
                                val n = input.read(buf); if (n < 0) break
                                output.write(buf, 0, n); done += n
                                onProgress((done.toFloat()/totalBytes).coerceIn(0f, 1f))
                            }
                        }
                    }
                }
            }
            dir
        }
}
