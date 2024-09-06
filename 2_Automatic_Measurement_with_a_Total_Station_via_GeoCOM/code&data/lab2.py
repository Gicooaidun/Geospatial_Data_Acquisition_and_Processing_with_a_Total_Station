# Description: Basic commands (few examples), Lab 2 for GSDAq, October 2023
# Chapter of Leica TPS1200 GeoCOM Reference Manual
# ASCII-Request
# ASCII-Response

# from datetime import datetime
import time
from time import sleep
import argparse
#pip install pyserial
import serial
import serial.tools.list_ports
import numpy as np
import pandas as pd
from measurement_processing import process_pandas
import datetime

PORT = 'COM3' # Needs to be set as input by user
ATR_mode = 1
ATR_window = 10 # gon
tolerances = [0.00063662, 0.00063662] # gon
prism_type = 0
target_type = 0
reflector_height = 0
Face = [1,1,0]


def write_log(message, filename):
    """
    This function writes a log entry with a timestamp to specific log file.

    @param
        message: str
            The log message to be written to the log file.
        filename: str
            The name of the log file (excluding the file extension).

    @return
        None

    """

    # Get the current timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create or open the log file in append mode
    with open('Output_log/' + filename  + '.log', "a") as file:
        log_entry = f"{timestamp} - {message}\n"
        file.write(log_entry)


def set_ATR_mode(TPS_port, filename_without_extension, params=[0]):
    """
    This function sets the status of the ATR mode using AUS_SetUserAtrState.

    @param
      TPS_port: str
          The TPS port to send the request to.
      params: int, optional
          Parameters for the ATR mode (default is 0).

    @return
      None
    """
    print("=====================================================")
    print("AUS_SetUserAtrState - setting the status of the ATR mode \n")
    write_log("AUS_SetUserAtrState - setting the status of the ATR mode \n",
                    filename_without_extension)
    request_id = "18005"
    RC, _ = interpolate(TPS_port, request_id, filename_without_extension, params)
    if int(RC) == 0:
        print('ATR mode successfully changed')
        write_log("ATR mode successfully changed",
                    filename_without_extension)
    else:
        write_log(f"GRC_Return-Code= {int(RC)}; Check the GRC return-code in manual! \n", filename_without_extension)
        raise ValueError(f"GRC_Return-Code= {int(RC)}; Check the GRC return-code in manual! \n")
    return None


def change_Face(TPS_port, filename_without_extension, params=Face):
    """
    This function turns the telescope to the other face.

    @param
      TPS_port: str
          The TPS port to send the request to.
      params: list, optional
          Additional parameters for turning the telescope (default is [0, 1, False]).
          - params[0]: int, optional
            Position mode: AUT_NORMAL = 0 or AUT_PRECISE = 1.
          - params[1]: int, optional
            Mode of ATR: AUT_POSITION = 0 or AUT_TARGET = 1
          - params[2]: bool, optional
            Reserved for future use. Always set to False

    @return
        None
    """
    print("=====================================================")
    print("AUT_ChangeFace - turning the telescope to the other face \n")
    write_log("AUT_ChangeFace - turning the telescope to the other face \n", filename_without_extension)
    request_id = "9028"
    RC, _ = interpolate(TPS_port, request_id, filename_without_extension, params)
    if int(RC) == 0:
        print('Sucessfully changed face of telescope')
        write_log('Sucessfully changed face of telescope', filename_without_extension)

    else:
        write_log(f"GRC_Return-Code= {int(RC)}; Check the GRC return-code in manual! \n", filename_without_extension)
        raise ValueError(f"GRC_Return-Code= {int(RC)}; Check the GRC return-code in manual! \n")
    return None


def set_ATR_window(TPS_port, filename_without_extension, params=[10]):
    """
    This function sets the ATR search window using AUT_SetUserSpiral.

    @param
        TPS_port: str
            The TPS port to send the request to.
        params: int, optional
            The size of the ATR search window (default is 10 gon).

    @return
        None
    """
    print("=====================================================")
    print("AUT_SetUserSpiral - setting the ATR search window \n")
    write_log("AUT_SetUserSpiral - setting the ATR search window \n", filename_without_extension)
    request_id = "9041"
    params[0] = gon2rad(params[0])
    RC, _ = interpolate(TPS_port, request_id, filename_without_extension, params)
    if int(RC) == 0:
        write_log("Sucessfully set up ATR window", filename_without_extension)
        print('Sucessfully set up ATR window')
    else:
        write_log(f"GRC_Return-Code= {int(RC)}; Check the GRC return-code in manual! \n", filename_without_extension)
        raise ValueError(f"GRC_Return-Code= {int(RC)}; Check the GRC return-code in manual! \n")
    return None

def set_tolerances(TPS_port, filename_without_extension, params=[0.00063662, 0.00063662]):
    """
    This function sets the positioning tolerances.

    @param
        TPS_port: str
            The TPS port to send the request to.
        params: list, optional
            The positioning tolerances in gonns (default is [0.00063662, 0.00063662]).
            - params[0]: float, optional
              Tolerance in Hz direction.
            - params[1]: float, optional
              Tolerance in V direction.

    @return
        None
    """
    print("=====================================================")
    print("AUT_SetTol - setting the positioning tolerances \n")
    write_log("AUT_SetTol - setting the positioning tolerances \n", filename_without_extension)
    request_id = "9007"
    params = [gon2rad(i) for i in params]
    RC, _ = interpolate(TPS_port, request_id, filename_without_extension, params)
    if int(RC) == 0:
        print('Sucessfully set the positioning tolerances')
        write_log('Sucessfully set the positioning tolerances', filename_without_extension)
    else:
        write_log(f"GRC_Return-Code= {int(RC)}; Check the GRC return-code in manual! \n", filename_without_extension)
        raise ValueError(f"GRC_Return-Code= {int(RC)}; Check the GRC return-code in manual! \n")
    return None 


def set_prsim_type(TPS_port, filename_without_extension, params=[0]):
    """
    This function sets the default prism type.

    @param
        TPS_port: str
            The TPS port to send the request to.
        params: int, optional
            The default prism type (default is 0).

    @return
        None
    """
    print("=====================================================")
    print("BAP_SetPrismType - setting the default prism type \n")
    write_log("BAP_SetPrismType - setting the default prism type \n", filename_without_extension)
    request_id = "17008"
    RC, _ = interpolate(TPS_port, request_id, filename_without_extension, params)
    if int(RC) == 0:
        print('Sucessfully set default prism type')
        write_log('Sucessfully set default prism type', filename_without_extension)
    else:
        write_log(f"GRC_Return-Code= {int(RC)}; Check the GRC return-code in manual! \n", filename_without_extension)
        raise ValueError(f"GRC_Return-Code= {int(RC)}; Check the GRC return-code in manual! \n")
    return None


def set_target_type(TPS_port, filename_without_extension, params=[0]):
    """
    This function sets the EDM type.

    @param
        TPS_port: str
            The TPS port to send the request to.
        params: int, optional
            The EDM type (reflector = 0, reflectorless = 1, default is 0).

    @return
        None
    """
    print("=====================================================")
    print("BAP_SetTargetType - setting the EDM type \n")
    request_id = "17021"
    RC, _ = interpolate(TPS_port, request_id, filename_without_extension, params)
    if int(RC) == 0:
        print('Sucessfully set EDM type')
    else:
        write_log(f"GRC_Return-Code= {int(RC)}; Check the GRC return-code in manual! \n", filename_without_extension)
        raise ValueError(f"GRC_Return-Code= {int(RC)}; Check the GRC return-code in manual! \n")
    return None


def set_distance_measurement(TPS_port, filename_without_extension, params=[6]):
    """
    This function sets the distance measurement program.

    @param
        TPS_port: str
            The TPS port to send the request to.
        params: int, optional
            The measurement program.

    @return
        None
    """
    print("=====================================================")
    print("BAP_SetMeasPrg - setting the distance measurement programm \n")
    write_log("BAP_SetMeasPrg - setting the distance measurement programm \n", filename_without_extension)
    request_id = "17019"
    RC, _ = interpolate(TPS_port, request_id, filename_without_extension, params)
    if int(RC) == 0:
        print('Sucessfully set distance meauserement programm')
        write_log('Sucessfully set distance meauserement programm', filename_without_extension)
    else:
        write_log(f"GRC_Return-Code= {int(RC)}; Check the GRC return-code in manual! \n", filename_without_extension)
        raise ValueError(f"GRC_Return-Code= {int(RC)}; Check the GRC return-code in manual! \n")
    return None



def set_height(TPS_port, filename_without_extension, params=[0]):
    """
    This function sets a new reflector height.

    @param
        TPS_port: str
            The TPS port to send the request to.
        params: float, optional
            The new reflector height (default is 0 meters).

    @return
        None
    """
    print("=====================================================")
    print("TMC_SetHeight - setting a new reflector height \n")
    write_log("TMC_SetHeight - setting a new reflector height \n", filename_without_extension)
    request_id = "2012"
    RC, _ = interpolate(TPS_port, request_id, filename_without_extension, params)
    if int(RC) == 0:
        print('Sucessfully changed reflector height')
        write_log('Sucessfully changed reflector height', filename_without_extension)
    else:
        write_log(f"GRC_Return-Code= {int(RC)}; Check the GRC return-code in manual! \n", filename_without_extension)
        raise ValueError(f"GRC_Return-Code= {int(RC)}; Check the GRC return-code in manual! \n")
    return None


def interpolate(TPS_port, request_id, filename_without_extension, params=''):
    """
    This function sends a request to a TPS device, processes the response, and extracts returned values.

    @param
        TPS_port: object
            The TPS port to send the request to.
        request_id: str
            The request ID used to form the request code, e.g., "2107".
        params: str, optional
            Additional parameters for the request (default is an empty string).

    @return
        tuple
            A tuple containing the return code (RC) as a string and a list of values.
    """

    # Add additional input parameters if provided
    param = ','.join([f'{a}' for a in params])
    # Generate the compelte request code, e.g., %R1Q2107:
    request = f"%R1Q,{request_id}:{param}\r\n".encode("ascii")
    print(f'Sending request: {request_id}')
    write_log(f'Sending request: {request_id}', filename_without_extension)
    # Request
    TPS_port.write(request)
    # Reply
    response = TPS_port.readline()
    # Extract the returned values
    header, parameters = response.split(b":", 1)
    assert header.split(b',')[0] == b'%R1P'
    parameters = parameters.rstrip()
    # Split the return code (RC)
    return_code, *values = parameters.split(b',')

    return return_code, values

def gon2rad(angle):
    """
    This function converts an angle in gon to radians.

    @param
        angle: float
            The angle in gon to be converted to radians.

    @return
        float
            The equivalent angle in radians.
    """
    return angle * np.pi / 200

def rad2gon(angle):
    """
    This function converts an angle in radians to gon (gradian).

    @param
        angle: float
            The angle in radians to be converted to gon.

    @return
        float
            The equivalent angle in gon.
    """
    return angle * 200 / np.pi

def measure_distance_angles(TPS_port, filename_without_extension):
    """
    This function sends a request to the TPS device to measure horizontal (Hz) and vertical (V) angles
    along with a single distance (Ds) and returns the results.

    @param
        TPS_port: object
            The TPS port to send the request to.

    @return
        tuple
            A tuple containing the horizontal (Hz) and vertical (V) angles in radians, and the distance (Ds) in meters.
    """
    print("=====================================================")
    print(('TMC_QuickDist - returning a slope distance and hz-angle, v-angle \n'))
    write_log('TMC_QuickDist - returning a slope distance and hz-angle, v-angle \n', filename_without_extension)
    request_id = "2117"
    
    RC, values = interpolate(TPS_port, request_id, filename_without_extension)
    if int(RC) == 0:
        Hz, V, Ds = rad2gon(float(values[0])), rad2gon(float(values[1])), float(values[2])
        write_log(f'Hz = {Hz :.4f} [gon]; V = {V :.4f} [gon]; Ds = {Ds :.4f} [m]', filename_without_extension)
        print(f'Hz = {Hz :.4f} [gon]; V = {V :.4f} [gon]; Ds = {Ds :.4f} [m]')
    elif int(RC) == 1284:
        Hz, V, Ds = rad2gon(float(values[0]), filename_without_extension), rad2gon(float(values[1]), filename_without_extension), float(values[2])
        write_log(f'Hz = {Hz :.4f} [gon]; V = {V :.4f} [gon]; Ds = {Ds :.4f} [m]', filename_without_extension)
        write_log("Accuracy not guaranteed", filename_without_extension)
        print(f'Hz = {Hz :.4f} [gon]; V = {V :.4f} [gon]; Ds = {Ds :.4f} [m]')
        print("Accuracy not guaranteed")
    else:
        write_log(f"GRC_Return-Code= {int(RC)}; Check the GRC return-code in manual! \n", filename_without_extension)
        raise ValueError(f"GRC_Return-Code= {int(RC)}; Check the GRC return-code in manual! \n")
    
    measure_tmc(TPS_port, filename_without_extension)
   
    return Hz, V, Ds

def measure_tmc(TPS_port, filename_without_extension, params=[0,0]):
    """
    This function is used to interupt the distance measurement so that it does not continuously measure.

    @param
        TPS_port: object
            The TPS port to send the request to.

    @return
        None
    """
    print("=====================================================")
    print(('TMC_DoMeasure - carrying out a distance measurement \n'))
    write_log('TMC_DoMeasure - carrying out a distance measurement \n', filename_without_extension)
    request_id = "2008"
    RC, values = interpolate(TPS_port, request_id, filename_without_extension, params=[0,0])
    if int(RC) == 0:
        print("Successfully interupted measure procedure")
        write_log("Successfully interupted measure procedure", filename_without_extension)
    else:
        write_log(f"GRC_Return-Code= {int(RC)}; Check the GRC return-code in manual! \n", filename_without_extension)
        raise ValueError(f"GRC_Return-Code= {int(RC)}; Check the GRC return-code in manual! \n")
    return None


def change_position(TPS_port, filename_without_extension, Hz_temp=300.0000, V_temp=100.0000):
    """
    This function turns the telescope to a specified position.

    @param
        TPS_port: object
            The TPS port to send the request to.
        Hz_temp: float, optional
            The horizontal (Hz) angle in gon to turn the telescope to (default is 300.0000 gon).
        V_temp: float, optional
            The vertical (V) angle in gon to turn the telescope to (default is 100.0000 gon).

    @return
        None
    """
    print("=====================================================")
    print("AUT_MakePositioning - turning the telescope to a specified position (Hz=%sgon, V=%sgon) \n")
    write_log("AUT_MakePositioning - turning the telescope to a specified position (Hz=%sgon, V=%sgon) \n", filename_without_extension)
    request_id = "9027"
    RC, _ = interpolate(TPS_port, request_id, filename_without_extension, [gon2rad(Hz_temp), gon2rad(V_temp), 0, 0])
    if int(RC) == 0:
        print(f"Telescope turned to Hz={Hz_temp :.4f} [gon] and V={V_temp :.4f} [gon]")
        write_log(f"Telescope turned to Hz={Hz_temp :.4f} [gon] and V={V_temp :.4f} [gon]", filename_without_extension)
    else:
        write_log(f"GRC_Return-Code= {int(RC)}; Check the GRC return-code in manual! \n", filename_without_extension)
        raise ValueError(f"GRC_Return-Code= {int(RC)}; Check the GRC return-code in manual! \n")
    return None


def main():
    # user input: file name
    filename_without_extension = input("Please enter file name for saving measurements and logs (without file extension):")
    write_log("Creation of Logfile: {}".format(filename_without_extension), filename_without_extension)

    # parse arguments
    args = parse_args()
    print("=====================================================")

    # Print out all available ports
    if args.print_all_ports:
        Ports_list = list(serial.tools.list_ports.comports())
        Ports = [list(Port)[0] for Port in Ports_list]
        print(f"All available ports: {Ports}\n")
        write_log(f"All available ports: {Ports}\n", filename_without_extension)

    # user input: port
    args.port = input("Please enter the correct port to establish the connection: ")
    write_log("Port (user input): {}".format(args.port), filename_without_extension)

    # Open TPS port if possible
    try:
    
        TPS_port = serial.Serial(port=args.port,
                                baudrate=args.baud_rate,
                                timeout=args.time_out,
                                )

        # check connection
        if TPS_port.isOpen():
            print("Port is connected")
            write_log("Port is connected", filename_without_extension)
        
    except Exception as e:
        print("")
        print("Most likely, the wrong port has been chosen. Error Message:")
        print(e)
        print("")
        write_log("Most likely, the wrong port has been chosen. Error Message:", filename_without_extension)
        write_log(e, filename_without_extension)


    # Configure the total station
    set_ATR_mode(TPS_port, filename_without_extension, params=[ATR_mode])
    set_ATR_window(TPS_port, filename_without_extension, params=[ATR_window])
    set_tolerances(TPS_port, filename_without_extension, params=tolerances)
    set_prsim_type(TPS_port, filename_without_extension, params=[prism_type])
    set_target_type(TPS_port, filename_without_extension, params=[target_type])
    set_height(TPS_port, filename_without_extension, params=[reflector_height])
    set_distance_measurement(TPS_port, filename_without_extension, params=[1])

    # automatic measurement of sets
    if args.loop_measure:
        measure_tmc(TPS_port, filename_without_extension, params=[0,0])

        # create dataframe for saving measurements
        df_measurements = pd.DataFrame(columns=['Point_ID', 'Hz_gon', 'V_gon', 'Ds_m', 'Face_nr', 'Set_nr'])

        # user input: number of points
        nr_points = int(input("How many points do you wish to measure? Please enter the number:"))
        write_log("Number of points to measure (user input): {}".format(nr_points), filename_without_extension)

        # user input: number of sets
        nr_sets = int(input("How many sets do you wish to measure? Please enter the number:"))
        write_log("Number of sets to measure (user input): {}".format(nr_sets), filename_without_extension)

        # iterate over points, manual measurement using total station
        for i in range(1, nr_points + 1):
            # user input: point id/name
            id = input("Please enter point ID/name of point nr {}:".format(i))

            # user input: continue script if total station approximately alligned to target
            input(
                "Please align the total station manually to point {}. Press 'Enter' to continue with the measurement:".format(
                    id))

            # measure angles and distance
            Hz, V, Ds = measure_distance_angles(TPS_port, filename_without_extension)


            # print measurement to log, save in dataframe
            meaurement = {'Point_ID': id, 'Hz_gon': Hz, 'V_gon': V, 'Ds_m': Ds, 'Face_nr': 1, 'Set_nr': 1}
            write_log("Measurement: {}".format(meaurement), filename_without_extension)
            df_measurements = pd.concat([df_measurements, pd.DataFrame([meaurement])], ignore_index=True)

        # reverse order of points for measurement in second Face
        point_locations = df_measurements.iloc[::-1].copy()
        
        # change Face of total station
        change_Face(TPS_port, filename_without_extension, params=Face)

        count = 0
        # iterate over points
        for index, row in point_locations.iterrows():
            # change position of total station

            if row['Hz_gon'] > 200:
                additional_constant_h = -200
            else:
                additional_constant_h = 200

            if row['V_gon'] > 200:
                additional_constant_v = -2*abs(200-row['V_gon'])
            else:
                additional_constant_v = 2*abs(200-row['V_gon'])



            if count !=0:
                change_position(TPS_port, filename_without_extension, Hz_temp=row['Hz_gon'] + additional_constant_h, V_temp=row['V_gon']+additional_constant_v)

            # measure distance and angles
            Hz, V, Ds = measure_distance_angles(TPS_port, filename_without_extension)

            # save measurements in log file and dataframe
            meaurement = {'Point_ID': row['Point_ID'], 'Hz_gon': Hz, 'V_gon': V, 'Ds_m': Ds, 'Face_nr': 2,
                            'Set_nr': 1}
            write_log("Measurement: {}".format(meaurement), filename_without_extension)
            df_measurements = pd.concat([df_measurements, pd.DataFrame([meaurement])], ignore_index=True)

            count +=1
        
        point_locations2 = point_locations.iloc[::-1].copy()
        df_measurements_set1 = pd.concat([point_locations2, point_locations], ignore_index=True)

        # iterate over sets if more than one set to measure
        if nr_sets > 1:
            for j in range(2, nr_sets + 1):
                # initialize Face number
                Face_nr = 1

                # count measurements in set
                count = 0
                
                # change Face of total station
                change_Face(TPS_port, filename_without_extension, params=Face)

                count_2 = 0

                # iterate over points in first set
                for index, row in df_measurements_set1.iterrows():
                    # move total station to target
                    additional_constant_h = 0
                    additional_constant_v = 0

                    if count >= nr_points:
                        if row['Hz_gon'] > 200:
                            additional_constant_h = -200
                        else:
                            additional_constant_h = 200

                        if row['V_gon'] > 200:
                            additional_constant_v = -2*abs(200-row['V_gon'])
                        else:
                            additional_constant_v = 2*abs(200-row['V_gon'])


                    hz_position = row['Hz_gon'] + additional_constant_h
        
                    v_position = row['V_gon'] + additional_constant_v

                    # switch to Face 2 if all points measured once
                    if count == nr_points:
                        Face_nr = 2
                        change_Face(TPS_port, filename_without_extension, params=Face)

                    if count !=0 and count != nr_points:
                        change_position(TPS_port, filename_without_extension, Hz_temp=hz_position, V_temp=v_position)

                    
                    

                    # measure distance
                    Hz, V, Ds = measure_distance_angles(TPS_port, filename_without_extension)

                    

                    # save measurements in log file and dataframe
                    meaurement = {'Point_ID': row['Point_ID'], 'Hz_gon': Hz, 'V_gon': V, 'Ds_m': Ds,
                                    'Face_nr': Face_nr,
                                    'Set_nr': j}
                    write_log("Measurement: {}".format(meaurement), filename_without_extension)
                    df_measurements = pd.concat([df_measurements, pd.DataFrame([meaurement])], ignore_index=True)

                    # count measurements
                    count += 1

                    count_2 +=1

        # print measurements into log file and save as csv
        print(df_measurements)
        df_measurements.to_csv('Measurements/{}.csv'.format(filename_without_extension), index=False)
        write_log("Measurements saved as csv. See Measurements/{}.csv".format(filename_without_extension),
                    filename_without_extension)

        # conduct ISO 17123-3 test procedure
        process_pandas(df_measurements, nr_sets, nr_points, filename_without_extension)
        write_log(
            "Results of measurements according to ISO 17123-3 saved as .log file. See Output_log/{}_iso_17123_3.log".format(
                filename_without_extension),
            filename_without_extension)

    # Close TPS port
    TPS_port.close()
    if not TPS_port.isOpen():
        print("=====================================================")
        print("Port is disconnected")
        print("=====================================================")
        write_log("Port is disconnected.", filename_without_extension)


    


def parse_args():
    """
    This function parses arguments for parameter setting.

    @return
        argparse.Namespace
            An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser("Parameter setting")
    parser.add_argument("--out_dir", type=str, default=f"./outputs/{time.strftime('%Y_%m_%d_%H:%M')}",
    help="set the path to save results")

    connect_arg = parser.add_argument_group(title="Connection command")
    connect_arg.add_argument('--print_all_ports', type=bool, default=True)
    connect_arg.add_argument("--port", default="COM1", type=str, help="select the correct port")
    connect_arg.add_argument("--baud_rate", default=9600, type=int)
    connect_arg.add_argument("--time_out", default=60, type=int)

    control_arg = parser.add_argument_group(title="Control command")
    control_arg.add_argument('--simple_angle_measurement', type=bool, default=False)
    control_arg.add_argument('--distance_angle_measurement', type=bool, default=False)
    control_arg.add_argument('--position_change', type=bool, default=False)
    control_arg.add_argument('--loop_measure', type=bool, default=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
