package deepDriver.dl.aml.distribution;

import java.util.HashMap;
import java.util.Map;

public class DistributionEnvCfg {
	static DistributionEnvCfg cfg = new DistributionEnvCfg();
	Map<String, Object> envHash = new HashMap<String, Object>();
	
	private DistributionEnvCfg() {		
	}

	public static DistributionEnvCfg getCfg() {
		return cfg;
	}

	public static void setCfg(DistributionEnvCfg cfg) {
		DistributionEnvCfg.cfg = cfg;
	}	
	
	public Object get(String key) {
		return envHash.get(key);
	}
	
	public int getInt(String key) {
		Object obj = get(key);
		if (obj == null) {
			return 0;
		}
		return (Integer) obj;
	}
	
	public String getString(String key) {
		return (String) envHash.get(key);
	}
	
	public void set(String key, Object obj) {
		envHash.put(key, obj);
	}
}
