package deepDriver.dl.aml.lstm.apps.util;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

public class GZIPUtil {
	
	private static Integer BUFF_SIZE = 1024;
	
	public static void compressFile(String inFile){
		try {
			compressFile(inFile, true);
		} catch (Exception e) {
			
		}
	}
	
    public static void compressFile(String inFile, boolean delete)  {
    	File f = new File(inFile);
        String outFile = inFile + ".gz";
        FileInputStream in = null;
        GZIPOutputStream out = null;
        try {
            in = new FileInputStream(f);
        }catch (FileNotFoundException e) {
        	System.out.println("File Not Found:" + inFile);
        	return;
        }
        try {
            out = new GZIPOutputStream(new FileOutputStream(outFile));
            byte[] buf = new byte[BUFF_SIZE];
            int len = 0;
            while ((len = in.read(buf, 0 ,BUFF_SIZE)) > 0) {
                out.write(buf, 0, len);
            }
            in.close();
            out.finish();
            out.flush();
            out.close();			
		} catch (Exception e) {

		}
        if (delete) {
        	f.delete();
        }
    }
    
    public static void decompressFile(String inName) {
    	try {
        	File f = new File(inName);
        	String outName = inName.replace(".gz", "");
        	File fout = new File(outName);
        	
        	if(fout.exists()){
        		System.out.println("Unzip File Already Exist:" + outName);
        		System.out.println("Rename:" + outName + "(1)");
        		fout = new File(outName + "(1)");
        	}
        	FileInputStream is = new FileInputStream(f);
        	FileOutputStream os = new FileOutputStream(fout);
        	GZIPInputStream gs = new GZIPInputStream(is);
        	int len;
            byte buf[] = new byte[BUFF_SIZE];  
            while ((len = gs.read(buf, 0, BUFF_SIZE)) != -1) {  
            	os.write(buf, 0, len);  
            }
            is.close();
            os.close();
            gs.close();			
		} catch (Exception e) {
			System.out.println(e);
		}
    }
    
    public static void main(String[] args){
    	//Compress File
    	String inputFile = "D:\\nlp\\corpus\\POS\\china_daily\\199801_dict.txt";
    	compressFile(inputFile);
    	
    	//Decompress File
    	//String gzFile = "D:\\nlp\\corpus\\POS\\china_daily\\199801_dict.txt.gz";
    	//decompressFile(gzFile);
    	
    }
	
}
