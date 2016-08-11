package deepDriver.dl.aml.distribution;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

public class Fs {
	
	String file;
	public static void writeObject2File(String file, Object obj) throws Exception {
		ObjectOutputStream oos = new ObjectOutputStream(
				new FileOutputStream(file));
		oos.writeUnshared(obj);
		oos.close();
	}
	
	public static void writeObj2FileWithTs(String file, Object obj) throws Exception {
		ObjectOutputStream oos = new ObjectOutputStream(
				new FileOutputStream(file));
		oos.writeUnshared(obj);
		oos.close();
	}
	
	public static Object readObjFromFile(String file) throws Exception {
		ObjectInputStream is = new ObjectInputStream(
				new FileInputStream(file));
		Object obj = is.readUnshared();
		is.close();
		return obj;
	}

}
