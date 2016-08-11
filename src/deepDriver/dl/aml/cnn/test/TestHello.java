package deepDriver.dl.aml.cnn.test;

import java.io.File;

import deepDriver.dl.aml.distribution.Fs;

public class TestHello {
	
	public static void read(File dir) throws Exception {
		String mfile = dir.getAbsolutePath()+"\\helloVo.m";
		HelloVo hv = (HelloVo) Fs.readObjFromFile(mfile);
		System.out.println("Read from "+mfile+", "+hv.name);
	}
	
	public static void save(File dir) throws Exception {
		HelloVo hv = new HelloVo();
		String mfile = dir.getAbsolutePath()+"\\helloVo.m";
		Fs.writeObject2File(mfile, hv);
		System.out.println("Save into "+mfile);
	}
	
	
	public static void main(String[] args) throws Exception {
		String sf = System.getProperty("user.dir");		
		File dir = new File(sf, "data");
		dir.mkdirs();
//		save(dir);
		read(dir);
	}

}
