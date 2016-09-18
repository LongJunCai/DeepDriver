package deepDriver.dl.aml.lstm.apps.util;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Properties;
import java.util.Set;

public class StringUtils {
	
	public static String joinString(List<String> list, String delimiter) {
		StringBuilder builder = new StringBuilder();
		if (list.size() > 0) {
			builder.append(list.get(0));
		}
		for (int i = 1; i < list.size(); i++) {
			builder.append(delimiter).append(list.get(i));
		}
		return builder.toString();
	}
	
	public static Properties parseArgs(String[] args) {
		Properties prop = new Properties();
		List<String> remainingArgs = new ArrayList<String>();
		if (args.length == 0) {
			return prop;
		} else {
			Set<Integer> index = new HashSet<Integer>();
			for (int i = 0; i < (args.length-1); i++) {
				String k1 = args[i];
				String k2 = args[i+1];
				if (!k1.isEmpty() && k1.charAt(0) == '-' 
						&& !k2.isEmpty() && k2.charAt(0) != '-'){
					k1 = k1.substring(1, k1.length()); //remove '-'
					prop.setProperty(k1, k2); //<K,V> flag, value
					index.add(i);
					index.add(i+1);
				}
			}
			for (int i = 0; i < args.length; i++) {
				String k3 = args[i];
				if (!k3.isEmpty() && k3.charAt(0) != '-'
						&& !index.contains(k3))
					remainingArgs.add(k3);
			}
		    if (!remainingArgs.isEmpty())
		    	prop.setProperty("OTHERS", joinString(remainingArgs, " "));
		}
		return prop;
	}
	
}
