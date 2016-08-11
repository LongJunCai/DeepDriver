package deepDriver.dl.aml.distribution;

public interface CommandFilter {
	
	public boolean filtCommand(String command);
	
	public CommandFilter nextCommandFilter();

}
