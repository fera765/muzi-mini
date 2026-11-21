package com.muzi.mini.engine

import android.app.ActivityManager
import android.content.Context

/**
 * Auto-config de hardware no boot (porte do nosso hwdetect.py).
 * Escolhe threads pelos nucleos de alto desempenho e aplica o guardrail de RAM.
 */
data class HwProfile(val cores: Int, val threads: Int, val totalRamMB: Long, val availRamMB: Long)

object HardwareConfig {
    const val RAM_LIMIT_MB = 500L     // guardrail do modelo em execucao

    fun detect(ctx: Context): HwProfile {
        val cores = Runtime.getRuntime().availableProcessors()
        // decode escala com nucleos; no Moto G60 (2 big + 6 little) usar ~metade
        // evita oversubscription dos little cores.
        val threads = (cores / 2).coerceAtLeast(2).coerceAtMost(cores)
        val am = ctx.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val mi = ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
        return HwProfile(cores, threads, mi.totalMem / 1_000_000, mi.availMem / 1_000_000)
    }

    /** Pode carregar este modelo sem violar a RAM disponivel? */
    fun fits(profile: HwProfile, modelApproxRamMB: Int): Boolean =
        modelApproxRamMB < profile.availRamMB - 200   // margem de SO/UI
}
