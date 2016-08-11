package deepDriver.dl.aml.cnn.test;

import java.io.Serializable;

public class HelloVo implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	/**
	 * 
	 */
	protected int [] type = {1, 2, 3};
	
	protected String name = "hello";
	
	String wd = "world";

	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}

	public String getWd() {
		return wd;
	}

	public void setWd(String wd) {
		this.wd = wd;
	}
	
	
}
