package deepDriver.dl.aml.distribution.test.cl;

import java.io.Serializable;

public class Employee implements Serializable {

	   /**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private int employeeNumber;
	   private String employeeName;

	   public Employee(int num, String name) {
	      employeeNumber = num;
	      employeeName= name;
	   }

	    public int getEmployeeNumber() {
	      return employeeNumber ;
	   }

	   public void setEmployeeNumber(int num) {
	      employeeNumber = num;
	   }

	   public String getEmployeeName() {
	      return employeeName ;
	   }

	   public void setEmployeeName(String name) {
	      employeeName = name;
	   }
	}
