package com.abrebo.tensorflowandroidapp;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.widget.Toast;

import com.abrebo.tensorflowandroidapp.databinding.ActivityMainBinding;
import com.abrebo.tensorflowandroidapp.ml.Model;
import com.abrebo.tensorflowandroidapp.ml.ModelUnquant;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;

public class MainActivity extends AppCompatActivity {
    private ActivityMainBinding binding;
    private Bitmap img;

    @SuppressLint("SetTextI18n")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        binding.buttonSec.setOnClickListener(view -> selectImage());
        binding.buttonTahmin.setOnClickListener(view -> processImage());
    }

    private void selectImage() {
        Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        intent.setType("image/*");
        startActivityForResult(intent, 10);
        binding.textView.setText("");
    }

    private void processImage() {
        if (img == null) {
            Toast.makeText(this, "Lütfen bir resim seçin.", Toast.LENGTH_SHORT).show();
            return;
        }

        img = Bitmap.createScaledBitmap(img, 224, 224, true);

        try {
            Model model = Model.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.UINT8);
            TensorImage tensorImage=new TensorImage(DataType.UINT8);
            tensorImage.load(img);
            ByteBuffer byteBuffer=tensorImage.getBuffer();
            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            Model.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
// Display the output results
            StringBuilder resultBuilder = new StringBuilder();
            for (int i = 0; i < outputFeature0.getFloatArray().length; i++) {
                resultBuilder.append("Result ").append(i).append(": ").append(outputFeature0.getFloatArray()[i]).append("\n");
            }
            binding.textView.setText(resultBuilder.toString());
            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == 10 && resultCode == RESULT_OK) {
            if (data != null) {
                Uri uri = data.getData();
                try {
                    img = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
                    binding.imageView.setImageBitmap(img);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }
}
