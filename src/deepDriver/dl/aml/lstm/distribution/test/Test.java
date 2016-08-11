package deepDriver.dl.aml.lstm.distribution.test;

public class Test {
	
	public static void main(String[] args) throws InterruptedException {
		double [][][] aa = new double[1000][][];
		for (int i = 0; i < aa.length; i++) {
			aa[i] = new double[6][];
			for (int j = 0; j < aa[i].length; j++) {
				aa[i][j] = new double[1000];
				for (int j2 = 0; j2 < aa.length; j2++) {
					aa[i][j][j2] = 0.1;
				}
			}
		}
		int cnt = 0;
		while (true) {			
			System.out.println(""+cnt ++);
			aa = new double[1000][][];
			for (int i = 0; i < aa.length; i++) {
				aa[i] = new double[6][];
				for (int j = 0; j < aa[i].length; j++) {
					aa[i][j] = new double[1000];
					for (int j2 = 0; j2 < aa.length; j2++) {
						aa[i][j][j2] = 0.1;
					}
				}
			}
//			Thread.sleep(100);
		}
	}

}
