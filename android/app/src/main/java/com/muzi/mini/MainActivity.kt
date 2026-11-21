package com.muzi.mini

import android.app.Application
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import com.muzi.mini.data.ModelCatalog
import com.muzi.mini.data.ModelDownloader
import com.muzi.mini.data.ModelSpec
import com.muzi.mini.engine.HardwareConfig
import com.muzi.mini.engine.HwProfile
import com.muzi.mini.engine.OnnxLlmEngine
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                val vm: AppViewModel = viewModel(
                    factory = object : ViewModelProvider.Factory {
                        @Suppress("UNCHECKED_CAST")
                        override fun <T : ViewModel> create(mc: Class<T>): T =
                            AppViewModel(application) as T
                    }
                )
                AppRoot(vm)
            }
        }
    }
}

class AppViewModel(app: Application) : AndroidViewModel(app) {
    val hw: HwProfile = HardwareConfig.detect(app)
    private val downloader = ModelDownloader(app.filesDir)
    private var engine: OnnxLlmEngine? = null

    var status by mutableStateOf("Pronto — ${hw.cores} nucleos, ${hw.availRamMB} MB livres")
    var progress by mutableStateOf<Float?>(null)
    var loadedModel by mutableStateOf<String?>(null)
    val messages = mutableStateListOf<Pair<String, String>>()  // (autor, texto)
    var generating by mutableStateOf(false)

    fun installed(spec: ModelSpec) = downloader.isInstalled(spec)
    fun fits(spec: ModelSpec) = HardwareConfig.fits(hw, spec.approxRamMB)

    fun downloadAndLoad(spec: ModelSpec) {
        if (spec.files.isEmpty()) { status = "Modelo ainda nao publicado."; return }
        viewModelScope.launch {
            try {
                status = "Baixando ${spec.label}..."
                val dir = downloader.download(spec) { progress = it }
                progress = null
                status = "Carregando no runtime..."
                engine?.close()
                engine = OnnxLlmEngine(hw.threads).also { it.load(dir.absolutePath) }
                loadedModel = spec.label
                status = "Modelo carregado. Contexto: ${spec.contextTokens} tokens" +
                        if (spec.ssm) " (SSM — 1M real)" else ""
            } catch (e: Exception) {
                progress = null; status = "Erro: ${e.message}"
            }
        }
    }

    fun send(text: String) {
        val eng = engine ?: run { status = "Carregue um modelo primeiro."; return }
        messages.add("voce" to text)
        messages.add("muzi" to "")
        generating = true
        viewModelScope.launch {
            try {
                val idx = messages.lastIndex
                val sb = StringBuilder()
                eng.generate(text).collect { piece ->
                    sb.append(piece); messages[idx] = "muzi" to sb.toString()
                }
            } catch (e: Exception) {
                messages[messages.lastIndex] = "muzi" to "[erro: ${e.message}]"
            } finally { generating = false }
        }
    }

    override fun onCleared() { engine?.close() }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AppRoot(vm: AppViewModel) {
    var tab by remember { mutableStateOf(0) }
    Scaffold(
        topBar = { CenterAlignedTopAppBar(title = { Text("Muzi Mini") }) },
        bottomBar = {
            NavigationBar {
                NavigationBarItem(selected = tab == 0, onClick = { tab = 0 },
                    icon = {}, label = { Text("Modelos") })
                NavigationBarItem(selected = tab == 1, onClick = { tab = 1 },
                    icon = {}, label = { Text("Chat") })
            }
        }
    ) { pad ->
        Box(Modifier.padding(pad)) {
            if (tab == 0) ModelsScreen(vm) else ChatScreen(vm)
        }
    }
}

@Composable
fun ModelsScreen(vm: AppViewModel) {
    Column(Modifier.fillMaxSize().padding(12.dp)) {
        Text(vm.status, style = MaterialTheme.typography.bodyMedium)
        vm.progress?.let { LinearProgressIndicator(progress = { it }, modifier = Modifier.fillMaxWidth().padding(vertical = 8.dp)) }
        Spacer(Modifier.height(8.dp))
        LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            items(ModelCatalog.models) { spec ->
                ElevatedCard {
                    Column(Modifier.padding(12.dp)) {
                        Text(spec.label, style = MaterialTheme.typography.titleSmall)
                        Text("${spec.paramsB} · ~${spec.approxRamMB} MB RAM · ctx ${spec.contextTokens}" +
                                if (spec.ssm) " · SSM 1M" else "",
                            style = MaterialTheme.typography.bodySmall)
                        if (!vm.fits(spec))
                            Text("Pode nao caber na RAM livre deste aparelho.",
                                color = MaterialTheme.colorScheme.error,
                                style = MaterialTheme.typography.bodySmall)
                        Spacer(Modifier.height(6.dp))
                        Button(onClick = { vm.downloadAndLoad(spec) },
                            enabled = spec.files.isNotEmpty() && vm.progress == null) {
                            Text(if (vm.installed(spec)) "Carregar" else "Baixar e carregar")
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun ChatScreen(vm: AppViewModel) {
    var input by remember { mutableStateOf("") }
    Column(Modifier.fillMaxSize().padding(12.dp)) {
        Text("Modelo: " + (vm.loadedModel ?: "nenhum"), style = MaterialTheme.typography.labelMedium)
        Column(Modifier.weight(1f).verticalScroll(rememberScrollState())) {
            vm.messages.forEach { (autor, texto) ->
                Text("$autor: $texto", Modifier.padding(vertical = 4.dp))
            }
        }
        Row(verticalAlignment = Alignment.CenterVertically) {
            OutlinedTextField(
                value = input, onValueChange = { input = it },
                modifier = Modifier.weight(1f), singleLine = false,
                keyboardOptions = KeyboardOptions.Default, placeholder = { Text("Mensagem") }
            )
            Spacer(Modifier.width(8.dp))
            Button(onClick = { if (input.isNotBlank()) { vm.send(input.trim()); input = "" } },
                enabled = !vm.generating) { Text("Enviar") }
        }
    }
}
