package deepDriver.dl.aml.distribution;

public class CommandFilterManager {
	CommandFilter root;
	public void filtCommand(String command) {
		filtCommand(root, command);
	}
	
	public void filtCommand(CommandFilter cf, String command) {
		if (!cf.filtCommand(command)) {
			filtCommand(cf.nextCommandFilter(), command);
		}
	}

}
