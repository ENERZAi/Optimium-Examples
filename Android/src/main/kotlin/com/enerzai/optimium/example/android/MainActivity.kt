package com.enerzai.optimium.example.android

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeContent
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.Button
import androidx.compose.material.Card
import androidx.compose.material.CircularProgressIndicator
import androidx.compose.material.MaterialTheme
import androidx.compose.material.Scaffold
import androidx.compose.material.SnackbarHost
import androidx.compose.material.SnackbarHostState
import androidx.compose.material.Text
import androidx.compose.material.TopAppBar
import androidx.compose.material.rememberScaffoldState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import androidx.core.content.ContextCompat
import androidx.lifecycle.viewmodel.compose.viewModel
import kotlinx.coroutines.launch
import org.opencv.android.OpenCVLoader

const val TAG = "optimium-android"
val DEFAULT_PADDING = 16.dp
val ITEMS = arrayOf("Preprocess", "Input", "Infer", "Output", "PostProcess")

class MainActivity : ComponentActivity() {
    companion object {
        val REQUIRED_PERMISSIONS = mutableListOf(
            Manifest.permission.CAMERA
        ).apply {
            if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P)
                add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
        }.toTypedArray()
    }

    init {
        if (!OpenCVLoader.initLocal())
            Log.d(TAG, "failed to load opencv")
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            MaterialTheme {
                val data: InferenceViewModel = viewModel(initializer = {
                    InferenceViewModel(applicationContext)
                })
                Content(data)
            }
        }
    }
}

fun checkPermissions(context: Context) = MainActivity.REQUIRED_PERMISSIONS.all {
    ContextCompat.checkSelfPermission(context, it) == PackageManager.PERMISSION_GRANTED
}

@Composable
fun Content(data: InferenceViewModel) {
    val scaffoldState = rememberScaffoldState()
    val snackbarHostState = remember { SnackbarHostState() }

    Scaffold(
        scaffoldState = scaffoldState,
        snackbarHost = { SnackbarHost(hostState = snackbarHostState) },
        modifier = Modifier.windowInsetsPadding(WindowInsets.safeContent),
        topBar = {
            TopAppBar(title = { Text("Optimium Example App") })
        }
    ) { innerPadding ->
        Box(
            modifier = Modifier
                .padding(innerPadding)
                .fillMaxSize()
        ) {
            when (data.uiState) {
                UiState.INIT -> {
                    Column(modifier = Modifier.align(Alignment.Center)) {
                        CircularProgressIndicator(
                            modifier = Modifier.width(64.dp)
                        )
                        Text("loading...")
                    }
                }

                UiState.ERROR -> {
                    Text("Error: ${data.cause ?: "unknown error"}")
                }

                else -> {
                    MainView(
                        snackbar = snackbarHostState,
                        data = data
                    )
                }
            }
        }
    }
}

fun getPhotoReq() = PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)

fun Map<String, Boolean>.accepted(): Boolean = all { it.value }

@Composable
fun MainView(snackbar: SnackbarHostState, data: InferenceViewModel) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val cameraLauncher =
        rememberLauncherForActivityResult(contract = ActivityResultContracts.TakePicture()) {
            if (!it) {
                Log.d(TAG, "cannot get image")
                data.cancel()
            } else {
                Log.d(TAG, "image found. start inference.")
                data.infer(data.imagePath)
            }
        }
    val photoLauncher =
        rememberLauncherForActivityResult(contract = ActivityResultContracts.PickVisualMedia()) {
            if (it == null) {
                data.cancel()
            } else {
                data.infer(it)
            }
        }

    val cameraPermissionLauncher =
        rememberLauncherForActivityResult(contract = ActivityResultContracts.RequestMultiplePermissions()) {
            if (it.accepted()) {
                data.createTempImage()
                cameraLauncher.launch(data.imagePath)
            } else {
                scope.launch {
                    snackbar.showSnackbar("Please grant permissions for camera.")
                }
            }
        }

    val photoPermissionLauncher =
        rememberLauncherForActivityResult(contract = ActivityResultContracts.RequestMultiplePermissions()) {
            if (it.accepted()) {
                photoLauncher.launch(getPhotoReq())
            } else {
                scope.launch {
                    snackbar.showSnackbar("Please grant permissions for photo.")
                }
            }
        }

    val scrollState = rememberScrollState()

    Column(Modifier.padding(DEFAULT_PADDING)) {
        Image(data.outputImage.asImageBitmap(), null,
            modifier = Modifier
                .fillMaxWidth()
                .clip(shape = RoundedCornerShape(15.dp))
                .background(Color.LightGray)
                .clickable {
                    data.selectData()
                }
        )
        Spacer(modifier = Modifier.padding(vertical = 10.dp))

        Column(modifier = Modifier.verticalScroll(scrollState)) {
            if (data.results.isEmpty()) {
                Text("Not executed or not detected.")
            } else {
                data.results.forEach {
                    Text("${it.box} / ${it.score}")
                }
            }

            ITEMS.forEach {
                val time = data.stats.get(it)

                if (time == null)
                    Text("$it: -")
                else
                    Text(String.format("%s: %.03fus", it, time))
            }
        }
    }

    when (data.uiState) {
        UiState.DATA_SELECTION -> {
            PhotoSelectionDialog(
                onDismiss = { data.cancel() },
                fromCamera = {
                    if (checkPermissions(context)) {
                        data.createTempImage()
                        cameraLauncher.launch(data.imagePath)
                    }
                    else
                        cameraPermissionLauncher.launch(MainActivity.REQUIRED_PERMISSIONS)
                },
                fromPhoto = {
                    if (checkPermissions(context)) {
                        photoLauncher.launch(getPhotoReq())
                    } else {
                        photoPermissionLauncher.launch(MainActivity.REQUIRED_PERMISSIONS)
                    }
                }
            )
        }

        UiState.INFER -> {
            InferWaitDialog()
        }

        else -> {}
    }
}

@Composable
fun InferWaitDialog() {
    Dialog(onDismissRequest = {}) {
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .height(200.dp)
                .padding(DEFAULT_PADDING),
            shape = RoundedCornerShape(16.dp)
        ) {
            Column(
                modifier = Modifier.fillMaxSize(),
                verticalArrangement = Arrangement.Center,
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                CircularProgressIndicator()
                Text("Running...")
            }
        }
    }
}

@Composable
fun PhotoSelectionDialog(onDismiss: () -> Unit, fromCamera: () -> Unit, fromPhoto: () -> Unit) {
    Dialog(onDismissRequest = { onDismiss() }) {
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .height(200.dp)
                .padding(DEFAULT_PADDING),
            shape = RoundedCornerShape(16.dp)
        ) {
            Column(
                modifier = Modifier.fillMaxSize(),
                verticalArrangement = Arrangement.Center,
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text("Where to get a photo?", Modifier.padding(DEFAULT_PADDING))

                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = DEFAULT_PADDING),
                    horizontalArrangement = Arrangement.Center
                ) {
                    Button(onClick = fromCamera) {
                        Text("From Camera")
                    }
                    Button(onClick = fromPhoto) {
                        Text("From Photo")
                    }
                }
            }
        }
    }
}


@Preview
@Composable
fun PreviewContent() {
//    Content(InferenceViewModel(android.test))
}