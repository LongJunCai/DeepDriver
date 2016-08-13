package deepDriver.dl.aml.cnn.img;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileOutputStream;

import javax.imageio.ImageIO;

//import com.sun.image.codec.jpeg.JPEGCodec;
//import com.sun.image.codec.jpeg.JPEGEncodeParam;
//import com.sun.image.codec.jpeg.JPEGImageEncoder;

public class Img2Matrix {
//	//Gray = 0.2989 * R + 0.5870 * G + 0.1140 * B
//	
//	public BufferedImage constructFixedSize(String srcFile, int tWidth, int tHeight) {
//		BufferedImage result = null;  
//		  
//        try {  
//            File srcfile = new File(srcFile);  
//            if (!srcfile.exists()) {  
//                System.out.println("文件不存在");  
//                  
//            }  
//            BufferedImage im = ImageIO.read(srcfile);   
//  
//            result = new BufferedImage(tWidth, tHeight,  
//                    BufferedImage.TYPE_INT_RGB);  
//  
//            result.getGraphics().drawImage(  
//                    im.getScaledInstance(tWidth, tWidth,  
//                            java.awt.Image.SCALE_SMOOTH), 0, 0, null);  
//              
//  
//        } catch (Exception e) {  
//            System.out.println("创建缩略图发生异常" + e.getMessage());  
//        }  
//          
//        return result;  
//	}
//	
//	
//    public boolean writeHighQuality(BufferedImage im, String fileFullPath) {  
//        try {  
//            FileOutputStream newimage = new FileOutputStream(fileFullPath);  
//            JPEGImageEncoder encoder = JPEGCodec.createJPEGEncoder(newimage);  
//            JPEGEncodeParam jep = JPEGCodec.getDefaultJPEGEncodeParam(im);  
//            jep.setQuality(0.9f, true);  
//            encoder.encode(im, jep);  
//            newimage.close();  
//            return true;  
//        } catch (Exception e) {  
//            return false;  
//        }  
//    } 
//    
//    public void getImagePixel(String image) throws Exception {  
//        int[] rgb = new int[3];  
//        File file = new File(image);  
//        BufferedImage bi = null;  
//        try {  
//            bi = ImageIO.read(file);  
//        } catch (Exception e) {  
//            e.printStackTrace();  
//        }  
//        int width = bi.getWidth();  
//        int height = bi.getHeight();  
//        int minx = bi.getMinX();  
//        int miny = bi.getMinY();  
//        System.out.println("width=" + width + ",height=" + height + ".");  
//        System.out.println("minx=" + minx + ",miniy=" + miny + ".");  
//        for (int i = minx; i < width; i++) {  
//            for (int j = miny; j < height; j++) {  
//                int pixel = bi.getRGB(i, j); // 下面三行代码将一个数字转换为RGB数字  
//                rgb[0] = (pixel & 0xff0000) >> 16;  
//                rgb[1] = (pixel & 0xff00) >> 8;  
//                rgb[2] = (pixel & 0xff);  
//                System.out.println("i=" + i + ",j=" + j + ":(" + rgb[0] + ","  
//                        + rgb[1] + "," + rgb[2] + ")");  
//                int gray = (int)(0.2989 * (double)rgb[2] + 0.5870 * (double)rgb[1] + 0.1140 * (double)rgb[0]);
//            }  
//        }  
//    }  
//
//
}
