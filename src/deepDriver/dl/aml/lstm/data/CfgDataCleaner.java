package deepDriver.dl.aml.lstm.data;

public class CfgDataCleaner {
	
	public static void clean(double [][][] data) {
		for (int i = 0; i < data.length; i++) {
			clean(data[i]);	
			data[i] = null;
		}
	}
	
	public static void clean(double [][] data) {
		for (int i = 0; i < data.length; i++) {
			data[i] = null;
		}
	}
	
	public void clean(double [] data) {
	}

}
