#!/usr/bin/env python3
# Cam positioner calibration (rotary/linear potentiometers)

from __future__ import print_function
import time
import epics
import sys
import numpy as np

from pprint import pprint
from collections import OrderedDict, namedtuple

from numpy import deg2rad, rad2deg, sin, cos
import scipy.optimize
import matplotlib.pyplot as plt


AxisInfo = namedtuple('AxisInfo', 'motor rotary_pot linear_pots')


class PV(epics.PV):
    def get(self, **kw):
        value = super(PV, self).get(**kw)
        if value is None:
            raise TimeoutError('Timed out while reading value')
        return value

    def get_several(self, num_readings, delay, **kw):
        'Return several readings'
        values = []
        for i in range(num_readings):
            values.append(self.get(**kw))
            time.sleep(delay)

        return values

    def get_averaged(self, num_readings, delay, **kw):
        'Get several readings, return average'
        values = self.get_several(num_readings, delay, **kw)
        return sum(values) / len(values)


epics.pv.PV = PV


class CamMotorAndPots(object):
    # Awful OO for my convenience

    def __init__(self, prefix, cam_number, linear_pot_format):
        info = self.axis_info[cam_number]

        self.prefix = prefix
        self.cam_number = cam_number
        self.info = info

        self.motor = epics.Motor(self.prefix + info.motor)
        self.stop_go_pv = self.motor.PV('SPMG', auto_monitor=False)
        self.stop_pv = self.motor.PV('STOP', auto_monitor=False)
        self.calibration_set_pv = self.motor.PV('SET', auto_monitor=False)
        self.setpoint_pv = self.motor.PV('VAL', auto_monitor=False)
        self.readback_pv = self.motor.PV('DRBV', auto_monitor=False)
        self.velocity_pv = self.motor.PV('VELO', auto_monitor=False)
        self.torque_enable_pv = self.motor.PV('CNEN', auto_monitor=False)
        self.rotary_pot_pv = PV(self.prefix + info.rotary_pot)
        self.linear_pot_pvs = [
            PV(self.prefix + linear_pot_format.format(pot_number))
            for pot_number in info.linear_pots
        ]

        self.all_pvs = [self.stop_go_pv, self.stop_pv, self.setpoint_pv,
                        self.readback_pv, self.velocity_pv, self.rotary_pot_pv,
                        self.torque_enable_pv, self.calibration_set_pv,
                        ] + self.linear_pot_pvs

        for pv in self.all_pvs:
            pv.wait_for_connection()

    @property
    def connected(self):
        return all(pv.connected for pv in self.all_pvs)

    def enable(self):
        self.torque_enable_pv.put(1, wait=True)
        self.stop_go_pv.put('Go', wait=True)

    def disable(self):
        self.torque_enable_pv.put(0, wait=True)
        self.stop_go_pv.put('Stop', wait=True)

    def move(self, pos):
        ret = self.motor.move(val=pos, dial=True, wait=True)
        if ret != 0:
            raise epics.motor.MotorException('Move failed: ret={}'.format(ret))

    def calibrate(self, position):
        assert self.connected

        self.stop_pv.put(1, wait=True)
        try:
            self.calibration_set_pv.put(1, wait=True)
            self.setpoint_pv.put(position, wait=True)
        finally:
            self.calibration_set_pv.put(0, wait=True)

    def __repr__(self):
        return ('<CamMotorAndPots cam_number={cam_number} prefix={prefix!r} '
                'connected={connected}>'
                ''.format(cam_number=self.cam_number,
                          prefix=self.prefix,
                          connected=self.connected)
                )


class HXUCamMotorAndPots(CamMotorAndPots):
    # Note: linear potentiometers are as follows for the LCLS-I girder:
    # girder potentiometer 1 = LP1-Y (CM1)
    # girder potentiometer 2 = LP2-Y (CM2/CM3)
    # girder potentiometer 3 = LP3-X (CM2/CM3)
    # girder potentiometer 5 = LP5-Y (CM4)
    # girder potentiometer 6 = LP6-Y (CM5)
    # girder potentiometer 7 = LP7-X (CM5)

    cam_to_pot_numbers = {
        1: (1, ),
        2: (2, 3),
        3: (2, 3),
        4: (5, ),
        5: (6, 7),
    }


HXUCamMotorAndPots.axis_info = OrderedDict(
    [(cam,
      AxisInfo(motor='CM{}MOTOR'.format(cam),
               rotary_pot='CM{}ADCM'.format(cam),
               linear_pots=HXUCamMotorAndPots.cam_to_pot_numbers[cam]))
     for cam in range(1, 6)
     ]
)


class SXUCamMotorAndPots(CamMotorAndPots):
    cam_to_pot_numbers = {
        1: ('LV3', ),
        2: ('LV1', 'LH1'),
        3: ('LV1', 'LH1'),
        4: ('LV2', 'LH2'),
        5: ('LV2', 'LH2'),
    }


SXUCamMotorAndPots.axis_info = OrderedDict(
    [(cam,
      AxisInfo(motor='CM{}MOTOR'.format(cam),
               rotary_pot='CM{}ADCM'.format(cam),
               linear_pots=SXUCamMotorAndPots.cam_to_pot_numbers[cam]))
     for cam in range(1, 6)
     ]
)


def move_through_range(cam, low=0, high=360, step=2):
    'Move a motor through its range, yielding at each position'
    for pos in range(low, high, step):
        cam.move(pos)
        yield pos


def check_connected(cams):
    for num, cam in cams.items():
        if not cam.connected:
            raise RuntimeError('All cams are not connected')


def get_all_linear_pots(cams):
    'All linear pots from the CamMotorAndPots dict'
    pots = {}

    for cam_num, cam in cams.items():
        for pot_num, pv in zip(cam.info.linear_pots, cam.linear_pot_pvs):
            pots[pot_num] = pv

    return pots


def load_data_from_file(fn):
    'Import a shell script which stores cam calibration data'
    with open(fn, 'rt') as f:
        lines = [line.strip() for line in f.readlines()]

    data = {'linear': {},
            'calibration': {}}

    name_map = {
        'gain_rmsFit': 'gain_rms_fit',
        'rotaryPotOffset': 'rotary_pot_offset',
        'linear_offset': 'linear_phase_offset',
    }

    for line in lines:
        line = line.strip()
        if not line:
            continue
        elif line.startswith('caput') or line.startswith('USEG'):
            line = line[line.index('CAL'):]
            items = [item for item in line.split(' ') if item]
            name = items[0]
            if name == 'CALVOLTAVG':
                data['calibration']['average_voltage'] = float(items[1])
            else:
                values = [float(item) for item in items[2:]]
                if name == 'CALCAMANGLE':
                    data['angles'] = values
                elif name == 'CALCAMPOT':
                    data['rotary'] = values
                else:
                    linear_idx = int(name[-1])
                    data['linear'][linear_idx] = values
                print(name, len(values))
        else:
            if line.startswith('# '):
                line = line[2:]

            if ' ' not in line:
                continue

            items = [item for item in line.split(' ') if item]
            name = items[0]
            if name.lower() != 'summary:':
                if name.lower() == 'cam_number':
                    data['cam'] = int(items[-1])
                else:
                    name = name_map.get(name, name)
                    data['calibration'][name] = float(items[-1])
            print(name, items[-1])

    return data


def get_calibration_data(cams, cam_num, velocity, dwell,
                         voltage_pv):
    'Move motors and get calibration data to be fit for a specific cam'
    check_connected(cams)
    voltage_pv.wait_for_connection()

    other_cams = [cams[other]
                  for other in {1, 2, 3, 4, 5} - {cam_num}
                  ]

    all_linear_pots = get_all_linear_pots(cams)
    for other in other_cams:
        other.disable()

    motor = cams[cam_num]
    motor.enable()
    motor.calibrate(0.0)
    motor.velocity_pv.put(velocity, wait=True)

    data = {'cam': cam_num,
            'angles': [],
            'rotary': [],
            'linear': {key: [] for key in all_linear_pots},
            'voltages': [],
            'calibration': {},
            }

    for pos in move_through_range(motor, 0, 360, 2):
        time.sleep(dwell)
        data['angles'].append(pos)
        data['rotary'].append(motor.rotary_pot_pv.get())
        data['voltages'].append(voltage_pv.get())

        for pot_num, linear_pot_pv in all_linear_pots.items():
            data['linear'][pot_num].append(linear_pot_pv.get())

    return data


def shift_for_polyfit(angles, rotary_pot, debug=False):
    # step = angles[1] - angles[0]
    # Find min/max rotary pot values, search for the deadband
    imin = np.argmin(rotary_pot)
    imax = np.argmax(rotary_pot)
    deadband_size = abs(imax - imin)
    deadband_shift = len(rotary_pot) - imin

    if debug:
        print('deadband size', deadband_size, file=sys.stderr)
        print('deadband shift', deadband_shift, file=sys.stderr)
    # Circularly shift rotary potentiometer data right for linear fitting
    rotary_pot = np.roll(rotary_pot, deadband_shift)

    # Linearly fit shifted rotary potentiometer data to compute gain
    ishiftmax = np.argmax(rotary_pot)
    return angles[:ishiftmax], rotary_pot[:ishiftmax]


def sinusoid(angles, amp, freq, phase, offset):
    'Function to be fit for the cams'
    return amp * sin(deg2rad(freq * angles + phase)) + offset


def sinusoid_params_from_linear_pot(angles, lin_pot):
    amp_guess = (np.max(lin_pot) - np.min(lin_pot)) / 2
    freq_guess = 1.

    rad_x = deg2rad(angles)
    lin_pot_transp = np.matrix(lin_pot).transpose()
    phase_rad = np.arctan2(cos(rad_x) * lin_pot_transp,
                           sin(rad_x) * lin_pot_transp)
    phase_guess = rad2deg(phase_rad)
    offset_guess = np.mean(lin_pot)
    return amp_guess, freq_guess, phase_guess[0, 0], offset_guess


def cam_sinusoidal_fit(angles, lin_pot, plot=False):
    'Perform sinusoidal fit on motor angles and linear pot data'
    angles = np.asarray(angles)
    lin_pot = np.asarray(lin_pot)

    guess = sinusoid_params_from_linear_pot(angles, lin_pot)
    amp_guess, freq_guess, phase_guess, offset_guess = guess

    if plot:
        plt.figure(0)

    def optimize_me(p):
        amp, freq, phase, offset = p
        opt = sinusoid(angles, amp, freq, phase, offset)
        if plot:
            plt.plot(opt)
        return np.sum((opt - lin_pot) ** 2)

    res = scipy.optimize.minimize(optimize_me, x0=guess, tol=0.0000001,
                                  options={'maxiter': 1000, 'disp': True})

    if plot:
        plt.legend()
        plt.show()

    lin_pot_fitted = sinusoid(angles, *res.x)

    amp, freq, phase, offset = res.x
    return (dict(amplitude=amp, frequency=freq, phase=phase, offset=offset),
            lin_pot_fitted)


def get_cam_to_pot_numbers(line):
    'hxr/sxr -> dictionary of cam motor to pot name/number'
    if line == 'hxr':
        return HXUCamMotorAndPots.cam_to_pot_numbers
    elif line == 'sxr':
        return SXUCamMotorAndPots.cam_to_pot_numbers
    raise ValueError('Unexpected line: {!r}; should be sxr or hxr', line)


def fit_data(data, line, plot=False):
    '''According to appropriate linear potentiometer, fit cam rotary pot'''
    cam_to_pot_numbers = get_cam_to_pot_numbers(line)
    cam_num = data['cam']
    angles = data['angles']
    rotary_pot = data['rotary']
    linear_pots = [data['linear'][num] for num in cam_to_pot_numbers[cam_num]]

    shifted_angles, shifted_rotary_pot = shift_for_polyfit(angles, rotary_pot)
    data['shifted_rotary'] = shifted_rotary_pot
    poly_rot = np.poly1d(np.polyfit(shifted_angles, shifted_rotary_pot, 1))

    yoff = poly_rot(np.asarray(shifted_angles))

    if plot:
        plt.clf()
        plt.title('Deadband shift')
        plt.plot(angles, rotary_pot, label='input')
        plt.plot(shifted_angles, shifted_rotary_pot, label='shifted')
        plt.plot(shifted_angles, yoff, label='post-fit')
        plt.legend()
        plt.plot()

    gain_rms_fit = np.std(shifted_rotary_pot - yoff)

    if 'voltages' in data:
        avg_voltage = np.average(data['voltages'])
        data['calibration']['average_voltage'] = avg_voltage
    else:
        avg_voltage = data['calibration']['average_voltage']

    gain = avg_voltage / poly_rot.coeffs[0]
    linear_pot = linear_pots[0]

    fit_result, linear_fitted = cam_sinusoidal_fit(angles, linear_pot)

    linear_phase_offset = fit_result['phase']

    if plot:
        plt.ion()
        plt.figure(cam_num)
        plt.clf()
        plt.plot(linear_pot, 'x', label='Data', lw=1)
        plt.plot(linear_fitted, label='Fitted')
        plt.legend()

    linear_offset_rms_fit = np.std(linear_pot - linear_fitted) * 2000.
    rotary_offset = (rotary_pot[0] / avg_voltage) * gain - linear_phase_offset
    return dict(average_voltage=avg_voltage,
                gain=gain,
                gain_rms_fit=gain_rms_fit,
                linear_phase_offset_rms_fit=linear_offset_rms_fit,
                linear_phase_offset=linear_phase_offset,
                rotary_pot_offset=rotary_offset,
                )


def compare_fits(data, labels, fit_info_dicts, line):
    cam_num = data['cam']
    angles = np.asarray(data['angles'])
    # rotary_pot = data['shifted_rotary']
    cam_to_pot_numbers = get_cam_to_pot_numbers(line)
    linear_pot = data['linear'][cam_to_pot_numbers[cam_num][0]]
    parameter_comparison = {
        'average_voltage': [],
        'gain': [],
        'gain_rms_fit': [],
        'linear_offset_rms_fit': [],
        'linear_phase_offset': [],
        'rotary_pot_offset': [],
    }

    plt.clf()

    plt.plot(angles, linear_pot, 'x', label='Linear pot')
    for idx, (fit_info, label) in enumerate(zip(fit_info_dicts, labels), 1):
        for key, value_list in parameter_comparison.items():
            value_list.append(fit_info.get(key, None))

        params = sinusoid_params_from_linear_pot(angles, linear_pot)
        amp_guess, freq_guess, phase_guess, offset_guess = params
        d = sinusoid(angles, amp_guess, freq_guess,
                     fit_info['linear_phase_offset'], offset_guess)
        plt.plot(angles, d, label=(label if label else 'Fit {}'.format(idx)))

    print('Parameter'.ljust(30, ' '), '\t'.join(labels))
    for key, values in parameter_comparison.items():
        vstr = '\t'.join('{:.5f}'.format(v) if v else 'None' for v in values)
        print(key.ljust(30, ' '), vstr)

    plt.legend()
    plt.show()


def setup_hgvpu(prefix='camsim:', linear_pot_format='LP{}ADCM'):
    cams = OrderedDict(
        (cam,
         HXUCamMotorAndPots(prefix, cam_number=cam,
                            linear_pot_format=linear_pot_format))
        for cam in HXUCamMotorAndPots.axis_info)
    return cams


def setup_sxu(prefix='camsim:', linear_pot_format='LP{}ADCM'):
    cams = OrderedDict(
        (cam,
         SXUCamMotorAndPots(prefix, cam_number=cam,
                            linear_pot_format=linear_pot_format))
        for cam in HXUCamMotorAndPots.axis_info)
    return cams


def write_data(f, data, segment='UND1:150', precision=5):
    prefix = 'USEG:{}:'.format(segment)
    name_map = OrderedDict(
        [('average_voltage', 'CALVOLTAVG'),
         ('angles', 'CALCAMANGLE'),
         ('rotary', 'CALCAMPOT'),
         ]
    )

    linear_pot_map = OrderedDict(
        [(1, 'CALGDRPOT1'),
         (2, 'CALGDRPOT2'),
         (3, 'CALGDRPOT3'),
         (5, 'CALGDRPOT5'),
         (6, 'CALGDRPOT6'),
         (7, 'CALGDRPOT7'),
         ]
    )

    def array_to_string(arr, precision):
        fmt = '{:.%df}' % precision
        return ' '.join(fmt.format(v) for v in arr)

    def write_array(f, name, value):
        print('{} {} {}'.format(prefix + name, len(value),
                                array_to_string(value, precision),
                                ),
              file=f)

    for name, write_name in name_map.items():
        if name == 'average_voltage':
            if 'calibration' not in data:
                continue
            value = data['calibration'][name]
            print('{} {}'.format(prefix + write_name, value), file=f)
        else:
            value = data[name]
            write_array(f, write_name, value)

    for pot_idx, write_name in linear_pot_map.items():
        write_array(f, write_name, data['linear'][pot_idx])

    print('', file=f)
    print('Summary:', file=f)
    print('cam_number = {}'.format(data['cam']), file=f)
    if 'calibration' in data:
        for key, value in sorted(data['calibration'].items()):
            fmt = '{:.%df}' % precision
            print('{} = {}'.format(key, fmt.format(value)), file=f)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    commands = parser.add_mutually_exclusive_group()
    commands.add_argument('--calibrate', type=str,
                          help='Calibrate motors with the given EPICS prefix')
    commands.add_argument('--load', type=str,
                          help='Load calibration data from file')

    parser.add_argument('--store',
                        action='store_true',
                        help='Store calibration data on motor')
    parser.add_argument('--save-to', type=str,
                        help='Save calibration data to file')

    parser.add_argument('-l', '--line', type=str, choices=('hxr', 'sxr'),
                        default='hgvpu', required=True,
                        help='Specify the undulator line')
    parser.add_argument('-s', '--segment', type=str,
                        help='Specify the undulator segment')
    parser.add_argument('-n', '--number', type=int,
                        help='Specify the cam positioner number')
    parser.add_argument('-p', '--plot',
                        action='store_true',
                        help='Plot relevant calibration information')
    parser.add_argument('--velocity', type=float, default=1.0,
                        help='Velocity for calibration')
    parser.add_argument('--voltage-pv', type=str, default='EXCTTNADCM',
                        help='Voltage PV suffix')
    parser.add_argument('--compare-to', type=str,
                        help='Compare calibration results with this file')
    parser.add_argument('--dwell',
                        type=float, default=1.0,
                        help='Dwell time after move')
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='Verbose operations')
    args = parser.parse_args()

    if args.load is not None:
        # 'data/2017-10-24/cam1_2017-10-24_10-12'
        data = load_data_from_file(args.load)
        if 'cam' not in data:
            if args.number is None:
                raise RuntimeError('Must specify the cam number')
            data['cam'] = args.number
        elif args.number is not None:
            data['cam'] = args.number
    elif args.calibrate:
        prefix = args.calibrate
        print('Connecting to {} line undulator ({}) with prefix {!r}'
              ''.format(args.line, args.segment, prefix))
        if args.line == 'hxr':
            motors = setup_hgvpu(prefix=prefix)
        elif args.line == 'sxr':
            motors = setup_sxu(prefix=prefix)
        else:
            raise ValueError('Unknown line; choose either sxr or hxr')
        voltage_pv = PV(prefix + args.voltage_pv, auto_monitor=False)

        print('Running calibration test...')
        data = get_calibration_data(motors, args.number,
                                    velocity=args.velocity, dwell=args.dwell,
                                    voltage_pv=voltage_pv)
        pprint(data)
        fit_results = fit_data(data, line=args.line)
        write_data(sys.stdout, data)
    else:
        print('Must specify either --load or --calibrate', file=sys.stderr)
        sys.exit(1)

    fit_results = fit_data(data, line=args.line)
    if args.save_to:
        with open(args.save_to, 'wt') as f:
            write_data(f, data)

    if args.compare_to:
        plt.figure(10)
        label1 = args.load if args.load else 'Calibrated'
        compare_fits(
            data,
            labels=[label1, args.compare_to],
            fit_info_dicts=[data['calibration'], fit_results],
            line=args.line,
        )
        plt.ioff()
        plt.show()
