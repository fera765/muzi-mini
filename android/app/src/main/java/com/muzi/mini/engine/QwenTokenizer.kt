package com.muzi.mini.engine

import org.json.JSONObject
import java.io.File

/**
 * Tokenizer byte-level BPE do Qwen2.5, lido de tokenizer.json.
 * Implementa o mesmo esquema do GPT-2/Qwen: pre-tokenizacao por regex,
 * mapeamento byte->unicode, merges BPE por rank. Aplica o chat template.
 */
class QwenTokenizer(tokenizerJson: File) {
    private val vocab = HashMap<String, Int>()
    private val idToTok = HashMap<Int, String>()
    private val ranks = HashMap<String, Int>()
    private val byteEncoder: Map<Int, Char>
    private val byteDecoder: Map<Char, Int>

    // especiais do Qwen2.5
    val imStart = 151644
    val imEnd = 151645
    val endOfText = 151643

    // GPT-2 regex de pre-tokenizacao
    private val pat = Regex(
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
    )

    init {
        val root = JSONObject(tokenizerJson.readText())
        val model = root.getJSONObject("model")
        val v = model.getJSONObject("vocab")
        for (k in v.keys()) { val id = v.getInt(k); vocab[k] = id; idToTok[id] = k }
        // added_tokens (especiais) tambem viram id<->tok para decode
        if (root.has("added_tokens")) {
            val at = root.getJSONArray("added_tokens")
            for (i in 0 until at.length()) {
                val o = at.getJSONObject(i)
                val id = o.getInt("id"); val c = o.getString("content")
                vocab[c] = id; idToTok[id] = c
            }
        }
        val merges = model.getJSONArray("merges")
        for (i in 0 until merges.length()) {
            // pode ser "a b" (string) ou ["a","b"] (array)
            val key = when (val m = merges.get(i)) {
                is String -> m
                else -> { val a = merges.getJSONArray(i); "${a.getString(0)} ${a.getString(1)}" }
            }
            ranks[key] = i
        }
        byteEncoder = bytesToUnicode()
        byteDecoder = byteEncoder.entries.associate { (b, c) -> c to b }
    }

    private fun bytesToUnicode(): Map<Int, Char> {
        val bs = ArrayList<Int>()
        (33..126).forEach { bs.add(it) }
        (161..172).forEach { bs.add(it) }
        (174..255).forEach { bs.add(it) }
        val cs = bs.toMutableList()
        var n = 0
        for (b in 0..255) if (b !in bs) { bs.add(b); cs.add(256 + n); n++ }
        return bs.indices.associate { bs[it] to cs[it].toChar() }
    }

    private fun bpe(token: String): List<String> {
        var word = token.map { it.toString() }.toMutableList()
        if (word.size < 2) return word
        while (true) {
            var best = Int.MAX_VALUE; var pos = -1
            for (i in 0 until word.size - 1) {
                val r = ranks["${word[i]} ${word[i + 1]}"] ?: continue
                if (r < best) { best = r; pos = i }
            }
            if (pos < 0) break
            val nw = ArrayList<String>(word.size)
            var i = 0
            while (i < word.size) {
                if (i == pos) { nw.add(word[i] + word[i + 1]); i += 2 }
                else { nw.add(word[i]); i++ }
            }
            word = nw
        }
        return word
    }

    private fun encodeOrdinary(text: String, out: ArrayList<Int>) {
        for (m in pat.findAll(text)) {
            val piece = m.value
            val mapped = buildString {
                for (b in piece.toByteArray(Charsets.UTF_8)) append(byteEncoder[b.toInt() and 0xFF])
            }
            for (t in bpe(mapped)) vocab[t]?.let { out.add(it) }
        }
    }

    /** Monta o prompt no chat template do Qwen e tokeniza. */
    fun encodeChat(userMsg: String, system: String = "You are a helpful assistant."): LongArray {
        val ids = ArrayList<Int>()
        fun role(tag: String, body: String) {
            ids.add(imStart); encodeOrdinary("$tag\n$body", ids); ids.add(imEnd); encodeOrdinary("\n", ids)
        }
        role("system", system)
        role("user", userMsg)
        ids.add(imStart); encodeOrdinary("assistant\n", ids)   // sem im_end: modelo continua
        return LongArray(ids.size) { ids[it].toLong() }
    }

    fun decode(ids: List<Int>): String {
        val sb = StringBuilder()
        for (id in ids) {
            if (id == imEnd || id == imStart || id == endOfText) continue
            sb.append(idToTok[id] ?: "")
        }
        val bytes = ArrayList<Byte>()
        for (c in sb) byteDecoder[c]?.let { bytes.add(it.toByte()) }
        return String(bytes.toByteArray(), Charsets.UTF_8)
    }

    fun decodeToken(id: Int): String {
        if (id == imEnd || id == imStart || id == endOfText) return ""
        val tok = idToTok[id] ?: return ""
        val bytes = ArrayList<Byte>()
        for (c in tok) byteDecoder[c]?.let { bytes.add(it.toByte()) }
        return String(bytes.toByteArray(), Charsets.UTF_8)
    }
}
