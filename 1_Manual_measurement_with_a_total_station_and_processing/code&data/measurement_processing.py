import numpy as np
import pandas as pd
import os

def read_file(path):
    """
    Reads and converts a GSI16 file into a numpy ndarray.
    
    @param 
        path: str
            Relative path to the GSI16 file.
    
    @return 
        data_array: numpy ndarray
            Array containing information from the GSI16 file.
    """
    with open(path,'r') as file:
        file.readline()
        data_list=[line.lstrip('*').split() for line in file]
    data_array = np.array(data_list)
    return data_array


def cal_num(data_array):
    """
    Calculate the number of sets and points based on the number of times the first point has been measured.
    
    @param 
        data_array: numpy ndarray
            Array containing information from the GSI16 file.
    
    @return 
        num_set: int
            Number of measured sets in the input file.
        num_point: int
            Number of distinct measured points.
    """
    # Get the value of the first point in the data_array
    first_value = data_array[0, 0]
    
    # Split the first_value using '+' as the delimiter and select the second part.
    result= first_value.split("+")[1]

    # Extract the first column of data_array
    data=data_array[:,0]

    # Loop through each measurement, count number of times a point appears in the file
    count=0
    for item in data:
        if result in item:
            count+=1

    # Get the total number of measurements
    num_mea=data_array.shape[0]

    # Get the number of sets (each point is measured in two phases)
    num_set=int(count/2)

    # Get the number of distinct points
    num_point=int(num_mea/count)
    
    return num_set,num_point


def arithmetic_check1(x_jk1,x_jk2,x_jk, num_set):
    """
    This function calculates the result of an arithmetic check based on the sums
    of measurements in two different directions and compares it to a threshold value.
    Based on the simplified test procedure for the horizontal directions of the ISO 17123-3.

    @param
        x_jk1: numpy ndarray
            Array containing measurements in direction 1.
        x_jk2: numpy ndarray 
            Array containing measurements in direction 2.
        x_jk: numpy ndarray 
            Array containing mean measurements of both directions.
        num_set: int
            Number of measured sets in the input file.

    @return
        bool: True if the arithmetic check is successful (within a threshold), otherwise False.
    """
    # Calculate the check values for each set
    check1=np.zeros(num_set)
    for j in range(0,num_set):
        check1[j]=(np.sum(x_jk1[j,:])+np.sum(x_jk2[j,:])-2*np.sum(x_jk[j,:]))%200

        # Check if the result is not equal to 0 and not within a small tolerance (0.001)
        if check1[j]!=0 and abs(check1[j]-200)>0.001:
            return False

    # If all checks pass, return True
    return True


def arithmetic_check2(x_jk,x_jk_slash, num_set, num_point):
    """
    This function calculates the result of a second arithmetic check.
    Based on the simplified test procedure for the horizontal directions of the ISO 17123-3.

    @param
        x_jk: numpy ndarray
            Array containing measurements in direction 1.
        x_jk_slash: numpy ndarray 
            Reduction of x_jk into the direction of the target No. 1.
        num_set: int
            Number of measured sets in the input file.
        num_point: int
            Number of distinct measured points.

    @return
        bool: True if the arithmetic check is successful (within a threshold), otherwise False.
    """
    # Calculate the check values for each set
    check2=np.zeros(num_set)
    for j in range(0,num_set):
        check2[j]=(np.sum(x_jk[j,:])-num_point*x_jk[j,0]-np.sum(x_jk_slash[j,:]))%400

        # Check if the result is not equal to 0 and not within a small tolerance (0.001)
        if check2[j]!=0 and abs(check2[j]-400)>0.001:
            return False

    # If all checks pass, return True
    return True


def arithmetic_check3(x_jk_slash,x_k_mean,d_jk, num_set):
    """
    This function calculates the result of a third arithmetic check.
    Based on the simplified test procedure for the horizontal directions of the ISO 17123-3.

    @param
        x_jk_slash: numpy ndarray 
            Reduction of x_jk into the direction of the target No. 1.
        x_k_mean: numpy ndarray
            Array containing the mean values of the directions x_jk.
        d_jk: numpy ndarray
            Array containing the differences of x_k_mean and x_jk_slash.
        num_set: int
            Number of measured sets in the input file.
    @return
        bool: True if the arithmetic check is successful (within a threshold), otherwise False.
    """
    # Calculate the check values for each set
    check3=np.zeros(num_set)
    for j in range(0,num_set):
        check3[j]=np.sum(x_k_mean)-np.sum(x_jk_slash[j,:])-np.sum(d_jk[j,:])

        # Check if the result is not within a very small tolerance (0.0000001)
        if abs(check3[j])>0.0000001:
            return False

    # If all checks pass, return True
    return True


def arithmetic_check4(r_jk, num_set):
    """
    This function calculates the result of a fourth arithmetic check.
    Based on the simplified test procedure for the horizontal directions of the ISO 17123-3.

    @param
        r_jk: numpy ndarray 
            Residuals of d_jk and d_j_mean.
        num_set: int
            Number of measured sets in the input file.
    @return
        bool: True if the arithmetic check is successful (within a threshold), otherwise False.
    """
    # Calculate the check values for each set
    check4=np.zeros(num_set)
    for j in range(0,num_set):
        check4[j]=np.sum(r_jk[j,:])

        # Check if the result is not within a very small tolerance (0.0000001)
        if abs(check4[j])>0.0000001:
            return False

    # If all checks pass, return True
    return True


def arithmetic_check5(x_jk_slash,x_k_mean,num_set):
    """
    This function calculates the result of a fifth arithmetic check.
    Based on the simplified test procedure for the horizontal directions of the ISO 17123-3.

    @param
        x_jk_slash: numpy ndarray 
            Reduction of x_jk into the direction of the target No. 1.
        x_k_mean: numpy ndarray 
            Array containing the mean values of the directions x_jk.
        num_set: int 
            Number of measured sets in the input file.
    @return
        bool: True if the arithmetic check is successful (within a threshold), otherwise False.
    """
    # Calculate the check value
    check5=np.sum(x_jk_slash)-num_set*np.sum(x_k_mean)

    # Check if the result is within a small tolerance (0.001)
    if abs(check5)<=0.001:
        return True
    else:
        return False


def arithmetic_check6(d_jk):
    """
    This function calculates the result of a sixth arithmetic check.
    Based on the simplified test procedure for the horizontal directions of the ISO 17123-3.

    @param
        d_jk: numpy ndarray
            Array containing the differences of x_k_mean and x_jk_slash.
    @return
        bool: True if the arithmetic check is successful (within a threshold), otherwise False.
    """
    # Calculate the check value
    check6=np.sum(d_jk)

    # Check if the result is within a small tolerance (0.001)
    if abs(check6)<=0.001:
        return True
    else:
        return False


def cal_sum_square_residuals(r_jk):
    """
    Calculate the sum of squares of measurement residuals.
    
    @param
        r_jk: numpy ndarray
            Residuals of d_jk and d_j_mean.
    @return
        r_square_sum: numpy float64
            Residuals sum of squares
    """
    r_square_sum=np.sum(np.square(r_jk))
    return r_square_sum


def cal_degree_of_freedom(num_set,num_point):
    """
    Calculates the degrees of freedom.
    
    @param
        num_set: int
            Number of measured sets in the input file.
        num_point: int
            Number of distinct measured points.
    @return
        v: int
            Degrees of freedom.
    """
    v=(num_set-1)*(num_point-1)
    return v


def cal_standard_deviation(r_square_sum,degree_of_freedom):
    """
    Calculate the standard deviation of the measurements.

    This function computes the standard deviation based on the sum of squares of residuals
    and the degrees of freedom.

    @param
        r_square_sum: numpy float64
            Residuals sum of squares
        degree_of_freedom: int
            Degrees of freedom.

    Returns:
        s: numpy float64
            Mesurements standard deviation.
    """
    s=np.sqrt(r_square_sum/degree_of_freedom)
    return s


def read_process_file(path):
    """
    This function reads data from a specified GSI16 file, processes it to calculate measurements and residuals and
    performs arithmetic checks on the measurements accodring to ISO 17123-3, calculates the standard deviation of 
    measurements and creates output log file. 

    @param
        path: str
            Relative path to the GSI16 file.

    The function performs the following steps:
    1. Read the file and convert it into a NumPy ndarray.
    2. Calculate the number of measured sets and distinct points in the data.
    3. Extract and encode horizontal angle measurements from the data.
    4. Calculate specific values (x_jk1, x_jk2, x_jk, etc.) based on the measurements.
    5. Perform arithmetic checks on the measurements using six separate check functions.
    7. Calculate the standard deviation of the measurements.
    8. Creates output log.
    """
    # Read file and convert it to ndarray
    data_array=read_file(path)

    # Get number of measured sets and distinct points
    num_set,num_point=cal_num(data_array)

    # Encode horizontal angles
    for i,element in enumerate(data_array[0,:]):
        if element.startswith('21'):
            column_index=i
    horizontal_angles=np.array([float(item.split("+")[1])/100000 for item in data_array[:,column_index]])

    # Calculate measurements and residuals table
    # Initalize data structures
    x_jk1=np.zeros((num_set,num_point))
    x_jk2=np.zeros((num_set,num_point))
    x_jk=np.zeros((num_set,num_point))
    x_jk_slash=np.zeros((num_set,num_point))
    x_k_mean=np.zeros(num_point)
    d_jk=np.zeros((num_set,num_point))
    d_j_mean=np.zeros(num_set)
    r_jk=np.zeros((num_set,num_point))

    # Calculate x_jk1, x_jk2 and x_jk
    for j in range(0,num_set):
        for k in range(0,num_point):
            x_jk1[j,k]=horizontal_angles[j*num_point*2+k]
            x_jk2[j,k]=horizontal_angles[j*num_point*2+num_point*2-k-1]
            if x_jk1[j,k]>200:
                x_jk[j,k]=(x_jk1[j,k]+x_jk2[j,k]+200)/2
            else:
                x_jk[j,k]=(x_jk1[j,k]+x_jk2[j,k]-200)/2
            x_jk_slash[j,k]=x_jk[j,k]-x_jk[j,0]

    # Calculate x_k_mean
    for k in range(0,num_point):
        x_k_mean[k]=np.sum(x_jk_slash[:,k])/num_set

    # Calculate d_jk and d_j_mean 
    for j in range(0,num_set):
        for k in range(0,num_point):
            d_jk[j,k]=x_k_mean[k]-x_jk_slash[j,k]
        d_j_mean[j]=np.sum(d_jk[j,:])/num_point

    # Calculate r_jk  
    for j in range(0,num_set):
        for k in range(0,num_point):
            r_jk[j,k]=d_jk[j,k]-d_j_mean[j]

    # Perform arithmetic checks
    measurements_correct = None
    if arithmetic_check1(x_jk1,x_jk2,x_jk,num_set) and arithmetic_check2(x_jk,x_jk_slash,num_set,num_point) and arithmetic_check3(x_jk_slash,x_k_mean,d_jk,num_set) and arithmetic_check4(r_jk, num_set) and arithmetic_check5(x_jk_slash,x_k_mean,num_set) and arithmetic_check6(d_jk):
        measurements_correct = 'The measurements are correct. All arithmetic checks were successful.'
    else:
        measurements_correct = 'The measurements are not correct. Not all arithmetic checks were successful.'

    # Calculate measurements standard deviation
    r_square_sum = cal_sum_square_residuals(r_jk)
    v = cal_degree_of_freedom(num_set,num_point)
    std = cal_standard_deviation(r_square_sum, v)

    # Print and save output logfile
    filename_without_extension = os.path.splitext(os.path.split(path)[1])[0]
    log_filename = 'Output_log/' + filename_without_extension + '.log'
    log_file = open(log_filename, 'w')

    print('This log contains the output log of the simplified test procedure (horizontal directions) according to ISO 17123-3.')
    log_file.write('This log contains the output log of the simplified test procedure (horizontal directions) according to ISO 17123-3.')

    print("")
    log_file.write('\n')
    log_file.write('\n')
    
    print('Input File: ' + str(os.path.split(path)[1]))
    log_file.write('Input File: ' + str(os.path.split(path)[1])+ '\n')

    print('Number of Measured Sets: ' + str(num_set))
    log_file.write('Number of Measured Sets: ' + str(num_set) + '\n')

    print('Number of Measured Points: ' + str(num_point))
    log_file.write('Number of Measured Points: ' + str(num_point) + '\n')

    print('Arithmetic Checks: ' + measurements_correct)
    log_file.write('Arithmetic Checks: ' + measurements_correct + '\n')

    print('Degrees of Freedom: ' + str(v))
    log_file.write('Degrees of Freedom: ' + str(v) + '\n')

    print('Residual Sum of Squares: ' + str(r_square_sum) + ' mgon\u00b2' )
    log_file.write('Residual Sum of Squares: ' + str(r_square_sum) + ' mgon\u00b2' + '\n')

    print('Measurements Standard Deviation: ' + str(std) + ' mgon')
    log_file.write('Measurements Standard Deviation: ' + str(std) + ' mgon' + '\n')
    
    log_file.write('This is an error message.\n')
    log_file.close()
    
    return

