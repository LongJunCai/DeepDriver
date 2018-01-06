package deepDriver.dl.aml.common.distribution;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.lang.reflect.Method;

import deepDriver.dl.aml.distribution.DistributionEnvCfg;
import deepDriver.dl.aml.distribution.P2PServer;
import deepDriver.dl.aml.distribution.ResourceMaster;
 

public class DistributionMaster {
	public static String MAIN = "-main="; 
	public static String JAR = "-jar="; 
	public static String SPLITTER = ";"; 
	
	public static void main(String[] args) throws Exception {
		/*****/
		DistributionEnvCfg.getCfg().set(P2PServer.KEY_SRV_PORT, 8034);
//		DistributionEnvCfg.getCfg().set(P2PServer.KEY_SRV_HOST, "127.0.0.1");
		ResourceMaster rm = ResourceMaster.getInstance();
		if (args != null && args.length > 2) {
			rm.setup(Integer.parseInt(args[0]));
		} else {
			rm.setup(4);
		}
		
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in, "utf-8"));
		while (true) { 
			String line = br.readLine();
			System.out.println("running with: "+line);
			String jarFile = commandParsing(line, JAR, SPLITTER);
			String mainClzz = commandParsing(line, MAIN, SPLITTER);
			System.out.println("Jar file is: "+ jarFile);
			System.out.println("mainClzz file is: "+ mainClzz);
			if (jarFile != null) {//load jar file.
//				ClassLoader.getSystemResource(jarFile).
			}
			
			Class clzz;
			try {
				clzz = Class.forName(mainClzz);
				Object obj = clzz.newInstance();
				Method m1 = clzz.getMethod("main", String[].class);//
				String [] aa = new String[]{};
				m1.invoke(null, (Object)aa);
			} catch (Exception e) {
	 			e.printStackTrace();
			}			
			
		}
		
	}
	
	public static String commandParsing(String line, String command, String splitter) {
		int pos = line.indexOf(command);
		if (pos >= 0) {
			String cmd = line.substring(pos+command.length());
			int split = cmd.indexOf(splitter);
			if (split < 0) {
				return null;
			}
			cmd = cmd.substring(0, split);
			return cmd;
		}
		return null;
	}

}
