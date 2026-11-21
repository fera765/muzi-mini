package com.muzi.mini.data

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.util.concurrent.TimeUnit

/**
 * Baixa os arquivos de um modelo com VERIFICACAO de tamanho (Content-Length).
 * Arquivos truncados (download cortado) sao detectados e re-baixados —
 * essencial em rede movel. Um .onnx incompleto quebra o ORT (protobuf).
 */
class ModelDownloader(private val baseDir: File) {
    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .build()

    fun modelDir(spec: ModelSpec) = File(baseDir, spec.id).apply { mkdirs() }

    fun isInstalled(spec: ModelSpec): Boolean {
        if (spec.files.isEmpty()) return false
        val dir = modelDir(spec)
        // "instalado" = todos os arquivos existem com tamanho plausivel (>1KB).
        // A verificacao real de tamanho acontece no download().
        return spec.files.all { File(dir, it.name).let { f -> f.exists() && f.length() > 1024 } }
    }

    suspend fun download(spec: ModelSpec, onProgress: (Float) -> Unit): File =
        withContext(Dispatchers.IO) {
            val dir = modelDir(spec)
            val totalApprox = (spec.files.sumOf { it.sizeMB.toLong() } * 1_000_000L).coerceAtLeast(1)
            var done = 0L
            for (f in spec.files) {
                val out = File(dir, f.name)
                val req = Request.Builder().url(f.url).build()
                client.newCall(req).execute().use { resp ->
                    if (!resp.isSuccessful) throw RuntimeException("Falha ao baixar ${f.name}: HTTP ${resp.code}")
                    val body = resp.body ?: throw RuntimeException("Resposta vazia em ${f.name}")
                    val expected = body.contentLength()   // -1 se desconhecido

                    // Ja completo? (tamanho bate com o servidor) -> pula sem rebaixar
                    if (out.exists() && expected > 0 && out.length() == expected) {
                        done += out.length(); onProgress((done.toFloat() / totalApprox).coerceIn(0f, 1f))
                        return@use
                    }

                    // (re)baixa inteiro
                    var written = 0L
                    body.byteStream().use { input ->
                        out.outputStream().use { output ->
                            val buf = ByteArray(1 shl 16)
                            while (true) {
                                val n = input.read(buf); if (n < 0) break
                                output.write(buf, 0, n); written += n; done += n
                                onProgress((done.toFloat() / totalApprox).coerceIn(0f, 1f))
                            }
                        }
                    }
                    // VERIFICA integridade: tamanho final deve bater
                    if (expected > 0 && out.length() != expected) {
                        out.delete()
                        throw RuntimeException("Download incompleto de ${f.name} (${out.length()}/$expected bytes). Tente de novo.")
                    }
                }
            }
            dir
        }
}
