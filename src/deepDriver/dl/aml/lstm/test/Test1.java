package deepDriver.dl.aml.lstm.test;

import java.util.Random;

public class Test1 {
	
	public static void main(String[] args) throws InterruptedException {
//		Map <Integer, Double> mp = new HashMap<Integer, Double>();
//		int a = 1;
//		double b = 2;
//		mp.put(a, b);
//		System.out.println(mp.get(a));
//		mp.put(a, mp.get(a) + 2);
//		System.out.println(mp.get(a));
		Random rd = new Random(10000);
		int k = 100;
		for (int i = 0; i < k; i++) {
			System.out.println(rd.nextDouble());
		}
		System.out.println("......");
		Thread.sleep(3000);
		Random rd1 = new Random(10000);
		for (int i = 0; i < k; i++) {
			System.out.println(rd1.nextDouble());
		}
	}

}
