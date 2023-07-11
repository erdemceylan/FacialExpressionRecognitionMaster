package com.pena.faceemotion;

import android.content.Context;
import android.content.SharedPreferences;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Paint;
import android.net.Uri;
import android.preference.PreferenceManager;
import android.provider.MediaStore;
import android.util.SparseArray;
import android.widget.Toast;

import com.google.android.gms.vision.Frame;
import com.google.android.gms.vision.face.Face;
import com.google.android.gms.vision.face.FaceDetector;
import com.pena.faceemotion.ml.MobilenetV2;
import com.pena.faceemotion.ml.MobilenetV22haziran;
import com.pena.faceemotion.ml.MobilenetV2Oversapmled;
import com.pena.faceemotion.roomData.Emotion;
import com.pena.faceemotion.roomData.EmotionDatabase;
import com.pena.faceemotion.roomData.IEmotionDAO;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class ImagesGallery {

    public static void listOfImages(Context context) {

        SharedPreferences sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context);
        long lastPhotoDate = sharedPreferences.getLong("last_photo_date", -1);

        String[] projection = {MediaStore.Images.Media.DATA, MediaStore.Images.Media.DATE_ADDED};
        String selection = null;
        String[] selectionArgs = null;
        String sortOrder = MediaStore.Images.Media.DATE_ADDED + " DESC";

        if (lastPhotoDate != -1) {
            selection = MediaStore.Images.Media.DATE_ADDED + ">?";
            selectionArgs = new String[]{String.valueOf(lastPhotoDate)};
        }

        Cursor cursor = context.getContentResolver().query(
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                projection,
                selection,
                selectionArgs,
                sortOrder
        );


        if (cursor != null && cursor.moveToFirst()) {
            do {
                String imagePath = cursor.getString(cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATA));
                long dateTaken = cursor.getLong(cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATE_ADDED));
                SharedPreferences.Editor editor = sharedPreferences.edit();
                editor.putLong("last_photo_date", dateTaken);
                editor.apply();

                final Bitmap myBitmap = BitmapFactory.decodeFile(imagePath);
                FaceDetector faceDetector = new FaceDetector.Builder(context.getApplicationContext())
                        .setTrackingEnabled(false)
                        .setLandmarkType(FaceDetector.ALL_LANDMARKS)
                        .setMode(FaceDetector.FAST_MODE)
                        .build();
                Frame frame = new Frame.Builder().setBitmap(myBitmap).build();
                SparseArray<Face> sparseArray = faceDetector.detect(frame);

                for (int i = 0; i < sparseArray.size(); i++) {
                    Face face = sparseArray.valueAt(i);

                    int g_x = (int) face.getPosition().x;
                    int g_y = (int) Math.max(0, face.getPosition().y); // Y değerini minimum 0 olarak sınırlayın
                    int width = (int) face.getWidth();
                    int height = (int) Math.min(myBitmap.getHeight() - g_y, face.getHeight()); // Yüksekliği geçerli boyuta sınırlayın

                    Bitmap faceBitmap = Bitmap.createBitmap(myBitmap, g_x, g_y, width, height);

                    Bitmap grayBitmap = grayscaleBitmap(faceBitmap);
                    int inputSize = 75; // Modelin beklediği giriş boyutu
                    Bitmap resizedBitmap = Bitmap.createScaledBitmap(grayBitmap, inputSize, inputSize, false);
                    ByteBuffer inputBuffer = ByteBuffer.allocateDirect(inputSize * inputSize * 3 * 4); // 3 kanal, 4 byte (float) kullanılıyor

                    inputBuffer.order(ByteOrder.nativeOrder()); // Byte sıralamasını düzenle

                    for (int y = 0; y < inputSize; y++) {
                        for (int x = 0; x < inputSize; x++) {
                            int pixel = resizedBitmap.getPixel(x, y);
                            // Piksel değerlerini 0-1 aralığına normalize et ve ByteBuffer'a ekle
                            inputBuffer.putFloat(((pixel >> 16) & 0xFF) / 255.0f); // Kırmızı kanal
                            inputBuffer.putFloat(((pixel >> 8) & 0xFF) / 255.0f); // Yeşil kanal
                            inputBuffer.putFloat((pixel & 0xFF) / 255.0f); // Mavi kanal
                        }
                    }

                    inputBuffer.rewind(); // ByteBuffer'ı başlangıç konumuna sıfırla
                    classifyEmotions(inputBuffer, context, imagePath);
                }
            } while (cursor.moveToNext());
            cursor.close();
        }
    }
    private static Bitmap grayscaleBitmap(Bitmap originalBitmap) {
        int width, height;
        height = originalBitmap.getHeight();
        width = originalBitmap.getWidth();

        Bitmap grayscaleBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(grayscaleBitmap);
        Paint paint = new Paint();
        ColorMatrix colorMatrix = new ColorMatrix();
        colorMatrix.setSaturation(0);
        ColorMatrixColorFilter colorMatrixFilter = new ColorMatrixColorFilter(colorMatrix);
        paint.setColorFilter(colorMatrixFilter);
        canvas.drawBitmap(originalBitmap, 0, 0, paint);

        return grayscaleBitmap;
    }


    private static void classifyEmotions(ByteBuffer byteBuffer, Context context, String ablosutePathOfImage) {
        EmotionDatabase emotionDatabase = EmotionDatabase.getEmotionDatabase(context);
        IEmotionDAO emotionDAO = emotionDatabase.getEmotionDAO();
        try {
            MobilenetV2 model = MobilenetV2.newInstance(context);

            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 75, 75, 3}, DataType.FLOAT32);
            inputFeature0.loadBuffer(byteBuffer);

            MobilenetV2.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            int maxPos = 0;
            float maxConfidence = 0;
            for (int i = 0; i < confidences.length; i++) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }

            String[] classes = {"Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"};

            Emotion emotion = new Emotion();
            emotion.setLabelName(classes[maxPos]);
            emotion.setImagePath(ablosutePathOfImage);
            if (emotionDAO.pathExists(emotion.getLabelName(), emotion.getImagePath()) != 1) {
                emotionDAO.insertEmotion(emotion);
            }
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }
}