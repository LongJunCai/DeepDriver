package deepDriver.dl.aml.contrib.MNIST;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

public class MnistLoader {
		
	private static String trainImg = "train-images-idx3-ubyte";
	private static String trainLable = "train-labels-idx1-ubyte";
	private static String testImg = "t10k-images-idx3-ubyte";
	private static String testLabel = "t10k-labels-idx1-ubyte";
	
	private static int TRAIN_SIZE = 60000;
	private static int TEST_SIZE = 10000;
	
	// double[][] Arrays Save Image Data
	private int[][] imgTrain;
	private int[][] imgTest;
	private int[] labelTrain;
	private int[] labelTest;
	
	public MnistLoader(String path) {
		try {
			read(path);
		} catch (IOException e) {
			System.out.println(e);
		}
	}
	
	@SuppressWarnings("resource")
	public int[][] readImage(String path, String file, Integer size)
			throws IOException {
		FileInputStream fileInputStream = new FileInputStream(new File(path
				+ "\\" + file));
		fileInputStream.skip(16);
		byte[][] imgByte = new byte[size][28 * 28];
		int[][] img = new int[size][28 * 28];
		for (int i = 0; i < size; i++) {
			fileInputStream.read(imgByte[i]);
			for (int j = 0; j < imgByte[0].length; j++) {
				// Byte(-126-127) to int(0-255)
				img[i][j] = (imgByte[i][j] & 0xff);
			}
		}
		return img;
	}
	
	@SuppressWarnings("resource")
	public int[] readLabel(String path, String file, int size)
			throws IOException {
		FileInputStream lableInputStream = new FileInputStream(new File(path
				+ "\\" + file));
		lableInputStream.skip(8);
		byte[] lablesByte = new byte[size];
		int[] lables = new int[size];
		lableInputStream.read(lablesByte);
		for (int i = 0; i < size; i++) {
			lables[i] = lablesByte[i] & 0xff;
		}
		return lables;
	}
	
	public void read(String path) throws IOException{
		imgTrain = readImage(path, trainImg, TRAIN_SIZE);
		imgTest = readImage(path, testImg, TEST_SIZE);
		labelTrain = readLabel(path, trainLable, TRAIN_SIZE);
		labelTest = readLabel(path, testLabel, TEST_SIZE);
		System.out.println("Image Train Size:" + imgTrain.length + "," + imgTrain[0].length);
		System.out.println("Image Test Size:" + imgTest.length + "," + imgTest[0].length);
		System.out.println("Label Train Size:" + labelTrain.length);
		System.out.println("Label Test Size:" + labelTest.length);
	}

	public int[][] getImgTrain() {
		return imgTrain;
	}

	public int[][] getImgTest() {
		return imgTest;
	}
	
	public int[] getLabelTrain() {
		return labelTrain;
	}

	public int[] getLabelTest() {
		return labelTest;
	}
	
}
