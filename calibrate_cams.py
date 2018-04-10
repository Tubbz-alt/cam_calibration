#!/usr/bin/env python3
# Cam positioner calibration (rotary/linear potentiometers)

from __future__ import print_function
import time
import epics
import os
import sys
import datetime
import numpy as np

from collections import OrderedDict, namedtuple
from contextlib import contextmanager

from numpy import deg2rad, rad2deg, sin, cos
import scipy.optimize
import matplotlib

try:
    matplotlib.use('Qt5Agg')
except Exception:
    pass

import matplotlib.pyplot as plt


voltage_suffix = 'EXCTTNADCM'
AxisInfo = namedtuple('AxisInfo',
                      'motor rotary_pot_adc rotary_pot_gain rotary_pot_offset '
                      'rotary_pot_calibrated linear_pots')

pyepics_move_codes = {
    -13: 'invalid value (cannot convert to float).  Move not attempted.',
    -12: 'target value outside soft limits.         Move not attempted.',
    -11: 'drive PV is not connected:                Move not attempted.',
    -8: 'move started, but timed-out.',
    -7: 'move started, timed-out, but appears done.',
    -5: 'move started, unexpected return value from PV.put()',
    -4: 'move-with-wait finished, soft limit violation seen',
    -3: 'move-with-wait finished, hard limit violation seen',
    0: 'move-with-wait finish OK / move-without-wait executed, not confirmed',
    1: 'move-without-wait executed, move confirmed',
    3: 'move-without-wait finished, hard limit violation seen',
    4: 'move-without-wait finished, soft limit violation seen',
}


@contextmanager
def set_soft_limits(cam, low_limit, high_limit, verbose=False):
    orig_llm = cam.llm_pv.get()
    orig_hlm = cam.hlm_pv.get()

    if verbose:
        print('Setting low limit to {}, high limit to {}'
              ''.format(low_limit, high_limit))

    cam.llm_pv.put(low_limit, wait=True)
    cam.hlm_pv.put(high_limit, wait=True)

    yield

    if verbose:
        print('Resetting low limit to {}, high limit to {}'
              ''.format(orig_llm, orig_hlm))

    cam.llm_pv.put(orig_llm, wait=True)
    cam.hlm_pv.put(orig_hlm, wait=True)


class PV(epics.PV):
    def __init__(self, pvname, auto_monitor=None, **kw):
        super(PV, self).__init__(pvname, auto_monitor=False, **kw)

    def get(self, use_monitor=False, **kw):
        value = super(PV, self).get(use_monitor=use_monitor, **kw)
        if value is None:
            raise TimeoutError('Timed out while reading value')
        return value


epics.pv.PV = PV


class CamMotorAndPots(object):
    # Awful OO for my convenience

    def __init__(self, prefix, cam_number, linear_pot_format='LP{}ADCM'):
        info = self.axis_info[cam_number]

        self.prefix = prefix
        self.cam_number = cam_number
        self.info = info

        try:
            self.motor = epics.Motor(self.prefix + info.motor)
        except TimeoutError:
            raise TimeoutError('Failed to connect to: {}'
                               ''.format(self.prefix + info.motor))

        self.stop_go_pv = self.motor.PV('SPMG')
        self.stop_pv = self.motor.PV('STOP')
        self.calibration_set_pv = self.motor.PV('SET')
        self.setpoint_pv = self.motor.PV('VAL')
        self.llm_pv = self.motor.PV('LLM')
        self.hlm_pv = self.motor.PV('HLM')
        self.readback_pv = self.motor.PV('RBV')
        self.velocity_pv = self.motor.PV('VELO')
        self.max_velocity_pv = self.motor.PV('VMAX')
        self.torque_enable_pv = self.motor.PV('CNEN')
        self.rotary_pot_pv = PV(self.prefix + info.rotary_pot_adc)
        self.calibrated_readback_pv = PV(self.prefix +
                                         info.rotary_pot_calibrated)
        self.rotary_pot_gain_pv = PV(self.prefix + info.rotary_pot_gain)
        self.rotary_pot_offset_pv = PV(self.prefix + info.rotary_pot_offset)

        self.linear_pot_pvs = [
            PV(self.prefix + linear_pot_format.format(pot_id))
            for pot_id in info.linear_pots
        ]

        self.all_pvs = [self.stop_go_pv, self.stop_pv, self.setpoint_pv,
                        self.llm_pv, self.hlm_pv, self.readback_pv,
                        self.velocity_pv, self.max_velocity_pv,
                        self.rotary_pot_pv, self.torque_enable_pv,
                        self.calibration_set_pv, self.calibrated_readback_pv,
                        ] + self.linear_pot_pvs

        for pv in self.all_pvs:
            pv.wait_for_connection()

    @property
    def connected(self):
        return all(pv.connected for pv in self.all_pvs)

    def enable(self):
        'Torque enable and SPMG=Go'
        self.torque_enable_pv.put(1, wait=True)
        self.stop_go_pv.put('Go', wait=True)

    def disable(self):
        'Torque disable and SPMG=Stop'
        self.torque_enable_pv.put(0, wait=True)
        self.stop_go_pv.put('Stop', wait=True)

    def normal_mode(self):
        'Torque disable and SPMG=Go'
        # In normal operation, we won't be poking single motors at a time, and
        # it's the responsibility of a higher level to enable torque when
        # starting motion.
        self.torque_enable_pv.put(0, wait=True)
        self.stop_go_pv.put('Go', wait=True)

    def move(self, pos):
        ret = self.motor.move(val=pos, wait=True)

        if ret != 0:
            raise epics.motor.MotorException(
                'Move to {} failed: ret={} ({})'
                ''.format(pos, ret, pyepics_move_codes.get(ret, '?'))
            )

    def calibrate_motor(self, position, verbose=False):
        assert self.connected

        with set_soft_limits(self, -360, 360, verbose=verbose):
            self.stop_pv.put(1, wait=True)
            try:
                self.calibration_set_pv.put(1, wait=True)
                self.setpoint_pv.put(position, wait=True)
            finally:
                self.calibration_set_pv.put(0, wait=True)

    def calibrate_rotary_pot(self, gain, offset):
        self.rotary_pot_gain_pv.put(gain, wait=True)
        self.rotary_pot_offset_pv.put(offset, wait=True)

    def __repr__(self):
        return ('<{class_name} cam_number={cam_number} prefix={prefix!r} '
                'connected={connected}>'
                ''.format(class_name=type(self).__name__,
                          cam_number=self.cam_number,
                          prefix=self.prefix,
                          connected=self.connected)
                )

    @classmethod
    def _get_axis_info(cls, cam_num):
        return AxisInfo(motor='CM{}MOTOR'.format(cam_num),
                        rotary_pot_adc='CM{}ADCM'.format(cam_num),
                        rotary_pot_calibrated='CM{}READDEG'.format(cam_num),
                        rotary_pot_gain='CM{}GAINC'.format(cam_num),
                        rotary_pot_offset='CM{}OFFSETC'.format(cam_num),
                        linear_pots=cls.cam_to_linear_pots[cam_num],
                        )


class HXUCamMotorAndPots(CamMotorAndPots):
    # Note: linear potentiometers are as follows for the LCLS-I girder:
    # girder potentiometer 1 = LP1-Y (CM1)
    # girder potentiometer 2 = LP2-Y (CM2/CM3)
    # girder potentiometer 3 = LP3-X (CM2/CM3)
    # girder potentiometer 5 = LP5-Y (CM4)
    # girder potentiometer 6 = LP6-Y (CM5)
    # girder potentiometer 7 = LP7-X (CM5)

    cam_to_linear_pots = {
        1: (1, ),
        2: (2, 3),
        3: (2, 3),
        4: (5, ),
        5: (6, 7),
    }


class SXUCamMotorAndPots(CamMotorAndPots):
    # even horizontal, odd vertical
    cam_to_linear_pots = {
        1: (3, ),
        2: (1, 2),
        3: (1, 2),
        4: (5, 4),
        5: (5, 4),
    }


line_to_class = {'hxr': HXUCamMotorAndPots,
                 'sxr': SXUCamMotorAndPots
                 }

for cls in (HXUCamMotorAndPots, SXUCamMotorAndPots):
    cls.axis_info = OrderedDict(
        [(cam, cls._get_axis_info(cam)) for cam in range(1, 6)]
    )


def move_through_range(cam, low=0, high=360, step=2):
    'Move a motor through its range, yielding at each position'
    for pos in range(low, high, step):
        cam.move(pos)
        yield pos


def check_connected(cams):
    for num, cam in cams.items():
        if not cam.connected:
            for pv in cam.all_pvs:
                if not pv.connected:
                    print('PV {} is not connected'.format(pv), file=sys.stderr)
            raise RuntimeError('All cams are not connected')


def get_all_linear_pots(cams):
    'All linear pots from the CamMotorAndPots dict'
    pots = {}

    for cam_num, cam in cams.items():
        for pot_id, pv in zip(cam.info.linear_pots, cam.linear_pot_pvs):
            pots[pot_id] = pv

    return pots


def load_data_from_file(fn, line):
    'Import a shell script which stores cam calibration data'
    with open(fn, 'rt') as f:
        lines = [lin.strip() for lin in f.readlines()]

    data = {'linear': {},
            'calibration': {}}

    name_map = {
        'gain_rmsFit': 'gain_rms_fit',
        'rotaryPotOffset': 'rotary_pot_offset',
        'linear_offset': 'linear_phase_offset',
    }

    in_summary = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        elif line.lower() == 'summary:':
            in_summary = True
        elif not in_summary:
            items = [item for item in line.split(' ') if item]
            name = items[0]
            if name == 'AVERAGE_INPUT_VOLTAGE':
                data['calibration']['average_input_voltage'] = float(items[1])
            else:
                values = [float(item) for item in items[2:]]
                if name in ('ANGLES', 'ROTARY'):
                    print('loaded ', name)
                    data[name.lower()] = values
                elif name.startswith('LINEAR_POT_'):
                    idx = int(name[-1])
                    data['linear'][idx] = values
                else:
                    raise ValueError('Unknown key: {}'.format(name))
                print(name, len(values))
        else:
            if line.startswith('# '):
                line = line[2:]

            if ' ' not in line:
                continue

            items = [item for item in line.split(' ') if item]
            name = items[0]

            if name.lower() == 'cam_number':
                data['cam'] = int(items[-1])
            elif name.lower() in ('prefix', 'line', 'serial'):
                data[name.lower()] = items[-1]
            elif name.lower() in ('passed', ):
                data[name.lower()] = (items[-1].lower() == 'true')
            else:
                name = name_map.get(name, name)
                data['calibration'][name] = float(items[-1])
            print(name, items[-1])

    return data


def get_calibration_data(cams, cam_num, velocity, dwell, voltage_pv,
                         verbose=False):
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
    orig_max_velocity = motor.max_velocity_pv.get()
    orig_velocity = motor.velocity_pv.get()

    data = {'cam': cam_num,
            'angles': [],
            'rotary': [],
            'linear': {key: [] for key in all_linear_pots},
            'voltages': [],
            'calibration': {},
            }

    try:
        motor.enable()
        motor.calibrate_motor(0.0)
        motor.max_velocity_pv.put(velocity, wait=True)
        motor.velocity_pv.put(velocity, wait=True)
        # extend soft limits

        with set_soft_limits(motor, -2, 362, verbose=verbose):
            for pos in move_through_range(motor, 0, 360, 2):
                if verbose and (pos % 20) == 0:
                    print('- Moved to {} degrees'.format(pos))
                time.sleep(dwell)
                data['angles'].append(pos)
                data['rotary'].append(motor.rotary_pot_pv.get())
                data['voltages'].append(voltage_pv.get())

                for pot_id, linear_pot_pv in all_linear_pots.items():
                    data['linear'][pot_id].append(linear_pot_pv.get())
            if verbose:
                print('Moving motor to 360 and setting position as 0 degrees')

            motor.move(360.0)
            motor.calibrate_motor(0.0)
    finally:
        if verbose:
            print('Resetting velocity to {}, max velocity to {}'
                  ''.format(orig_velocity, orig_max_velocity))
        motor.max_velocity_pv.put(orig_max_velocity, wait=True)
        motor.velocity_pv.put(orig_velocity, wait=True)

        if verbose:
            print('Setting cam motors back to normal operation mode')

        motor.normal_mode()
        for other in other_cams:
            other.normal_mode()

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


def get_cam_to_linear_pots(line):
    'hxr/sxr -> dictionary of cam motor to pot name/number'
    try:
        cls = line_to_class[line]
    except KeyError:
        raise ValueError('Unexpected line: {!r}; should be sxr or hxr', line)

    return cls.cam_to_linear_pots


def twin_legend(ax1, ax2, **kw):
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, **kw)


def check_pass_fail(rotary_pot, linear_pot):
    rotary_peak_idx = np.argmax(rotary_pot)

    start_idx, points = rotary_peak_idx + -1, 10
    slope_check_lin = linear_pot[start_idx:start_idx + points]
    slope, yint = np.polyfit(range(points), slope_check_lin, 1)
    return (start_idx, points), slope < 0


def fit_data(data, line, plot=False, verbose=False):
    '''According to appropriate linear potentiometer, fit cam rotary pot'''
    cam_num = data['cam']
    angles = data['angles']
    rotary_pot = data['rotary']

    cam_to_linear_pots = get_cam_to_linear_pots(line)
    linear_pot_numbers = cam_to_linear_pots[cam_num]
    linear_pot_number = linear_pot_numbers[0]
    linear_pots = [data['linear'][num] for num in linear_pot_numbers]

    shifted_angles, shifted_rotary_pot = shift_for_polyfit(angles, rotary_pot)
    data['shifted_rotary'] = shifted_rotary_pot
    poly_rot = np.poly1d(np.polyfit(shifted_angles, shifted_rotary_pot, 1))

    yoff = poly_rot(np.asarray(shifted_angles))

    gain_rms_fit = np.std(shifted_rotary_pot - yoff)

    if 'voltages' in data:
        avg_voltage = np.average(data['voltages'])
        data['calibration']['average_input_voltage'] = avg_voltage
    else:
        avg_voltage = data['calibration']['average_input_voltage']

    gain = avg_voltage / poly_rot.coeffs[0]
    linear_pot = linear_pots[0]

    fit_result, linear_fitted = cam_sinusoidal_fit(angles, linear_pot)

    linear_phase_offset = fit_result['phase']
    # NOTE: octave includes a factor of 2000 below, which we removed
    linear_offset_rms_fit = np.std(linear_pot - linear_fitted)
    rotary_offset = ((rotary_pot[0] / avg_voltage) * gain -
                     linear_phase_offset - 180)

    def shift_180(d):
        return np.roll(d, len(d) // 2)

    slope_check_info, passed = check_pass_fail(rotary_pot, linear_pot)

    try:
        fig, ax = plt.subplots(1, 1, figsize=(9, 6))
        plt.title('Cam {} Calibration'.format(data['cam']))
        plt.xlabel('Angle [deg]')
        plt.ylabel('Linear potentiometer [V]')
        angles = np.asarray(angles) - 180
        ax.plot(angles, shift_180(linear_pot), 'o', markersize=0.5,
                label='Calibration pot ({})'.format(linear_pot_number), lw=1)
        ax.plot(angles, shift_180(linear_fitted),
                label='Fitted calibration pot')

        start_idx, points = slope_check_info

        checks = np.zeros_like(linear_pot)
        checks[start_idx:start_idx + points] = linear_pot[start_idx:
                                                          start_idx + points]
        checks = shift_180(checks)

        ax.plot(angles[checks != 0], checks[checks != 0], 'x',
                label='Linear pot check area')

        ax.set_xlim(angles[0], angles[-1])

        if verbose:
            for pot, pot_values in data['linear'].items():
                if pot != linear_pot_number:
                    ax.plot(angles, shift_180(pot_values),
                            label='Calibration pot ({})'.format(pot),
                            lw=0.5)

        twin_ax = ax.twinx()
        twin_ax.set_ylabel('Rotary potentiometer [V]')
        twin_ax.plot(angles, shift_180(rotary_pot), label='Rotary pot',
                     color='indigo')
        twin_ax.plot(angles + rotary_offset, shift_180(rotary_pot),
                     alpha=0.4, color='lightblue', label='Rotary pot shifted')
        twin_legend(ax, twin_ax, loc='upper right')

        text_info = '''
Status           : {}
Rotary gain      : {:.4f}
Rotary offset    : {:.4f}
----------------------------
Gain fit RMS     : {:.4f}
Phase offset     : {:.4f}
Linear fit RMS   : {:.4f}
'''.format('PASSED' if passed else '**FAILED**',
           gain, rotary_offset, gain_rms_fit, linear_phase_offset,
           linear_offset_rms_fit)

        plt.annotate(text_info, xy=(0.72, 0.01),
                     xycoords='axes fraction',
                     family='monospace',
                     va="bottom",
                     fontsize=8)
    except Exception as ex:
        print('ERROR: Plotting failed {}: {}'
              ''.format(type(ex).__name__, ex))
    else:
        if plot:
            plt.plot()

    return dict(average_input_voltage=avg_voltage,
                gain=gain,
                gain_rms_fit=gain_rms_fit,
                linear_phase_offset_rms_fit=linear_offset_rms_fit,
                linear_phase_offset=linear_phase_offset,
                rotary_pot_offset=rotary_offset,
                passed=passed,
                )


def compare_fits(data, labels, fit_info_dicts, line):
    cam_num = data['cam']
    angles = np.asarray(data['angles'])
    # rotary_pot = data['shifted_rotary']
    cam_to_linear_pots = get_cam_to_linear_pots(line)
    linear_pot = data['linear'][cam_to_linear_pots[cam_num][0]]
    parameter_comparison = {
        'average_input_voltage': [],
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


def setup_hgvpu(prefix):
    cams = OrderedDict(
        (cam, HXUCamMotorAndPots(prefix, cam_number=cam))
        for cam in HXUCamMotorAndPots.axis_info)
    return cams


def setup_sxu(prefix):
    cams = OrderedDict(
        (cam, SXUCamMotorAndPots(prefix, cam_number=cam))
        for cam in HXUCamMotorAndPots.axis_info)
    return cams


def write_data(f, data, prefix, line, serial, precision=5):
    def array_to_string(arr, precision):
        fmt = '{:.%df}' % precision
        return ' '.join(fmt.format(v) for v in arr)

    def write_array(f, name, value):
        print('{} {} {}'.format(name, len(value),
                                array_to_string(value, precision),
                                ),
              file=f)

    for name in ('average_input_voltage', 'angles', 'rotary'):
        write_name = name.upper()
        if name == 'average_input_voltage':
            if 'calibration' not in data:
                continue
            value = data['calibration'][name]
            print('{} {}'.format(write_name, value), file=f)
        else:
            value = data[name]
            write_array(f, write_name, value)

    for pot_idx, pot_data in data['linear'].items():
        write_array(f, 'LINEAR_POT_{}'.format(pot_idx), pot_data)

    print('', file=f)
    print('Summary:', file=f)
    if 'calibration' in data and 'passed' in data['calibration']:
        print('passed = {}'.format(data['calibration']['passed']), file=f)

    print('line = {}'.format(line), file=f)
    print('serial = {}'.format(serial), file=f)
    print('cam_number = {}'.format(data['cam']), file=f)
    print('prefix = {}'.format(prefix), file=f)

    if 'calibration' in data:
        fmt = '{:.%df}' % precision
        for key, value in sorted(data['calibration'].items()):
            if key not in ('passed', ):
                print('{} = {}'.format(key, fmt.format(value)), file=f)


def main(args):
    if args.load is not None:
        # 'data/2017-10-24/cam1_2017-10-24_10-12'
        data = load_data_from_file(args.load, line=args.line)
        if 'cam' not in data:
            if args.number is None:
                raise RuntimeError('Must specify the cam number')
            data['cam'] = args.number
        elif args.number is not None:
            data['cam'] = args.number
    elif args.calibrate:
        if args.number is None:
            print('ERROR: Must specify cam positioner number to calibrate')
            sys.exit(1)

        prefix = args.calibrate
        print('Connecting to {} line undulator (serial {}) with prefix {!r}'
              ''.format(args.line, args.serial, prefix))
        if args.line == 'hxr':
            motors = setup_hgvpu(prefix=prefix)
        elif args.line == 'sxr':
            motors = setup_sxu(prefix=prefix)
        else:
            raise ValueError('Unknown line; choose either sxr or hxr')

        voltage_pv = PV(prefix + voltage_suffix)

        if args.verbose:
            def print_connected(pv):
                print('{}\t{}' ''.format(pv.pvname, 'connected'
                                         if pv.connected
                                         else 'disconnected'),
                      file=sys.stderr)

            for num, motor in motors.items():
                print('-- cam {} / {} --'.format(num, motor), file=sys.stderr)
                for pv in motor.all_pvs:
                    print_connected(pv)
                print(file=sys.stderr)
                print(file=sys.stderr)

            voltage_pv.wait_for_connection()
            print_connected(voltage_pv)

        print('Running calibration test on cam {}...'.format(args.number))
        data = get_calibration_data(motors, args.number,
                                    velocity=args.velocity, dwell=args.dwell,
                                    voltage_pv=voltage_pv,
                                    verbose=args.verbose)
        data['prefix'] = prefix

    fit_results = fit_data(data, line=args.line, plot=args.plot,
                           verbose=args.verbose)
    data['calibration'] = fit_results

    if args.save_to:
        with open(args.save_to, 'wt') as f:
            write_data(f, data, prefix=data['prefix'], line=args.line,
                       serial=args.serial)
        plt.savefig('{}.pdf'.format(args.save_to))
    elif args.calibrate:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        fn = os.path.join('data',
                          '{}_cam{}_{}.txt'.format(args.serial, data['cam'],
                                                   timestamp))
        try:
            os.makedirs('data')
        except Exception:
            pass

        try:
            with open(fn, 'wt') as f:
                write_data(f, data, prefix=data['prefix'], line=args.line,
                           serial=args.serial)
        except Exception as ex:
            print('Failed to save results to {}: {} {}'
                  ''.format(fn, type(ex).__name__, ex))
        else:
            plt.savefig('{}.pdf'.format(fn))
            print('Saved results to {}'.format(fn))

    if args.verbose:
        write_data(sys.stdout, data, prefix=data['prefix'], line=args.line,
                   serial=args.serial)

    if args.store_to_pv and fit_results.get('passed'):
        cam = motors[args.number]
        cam.calibrate_rotary_pot(fit_results['gain'],
                                 fit_results['rotary_pot_offset'])

        calibrated_position = cam.calibrated_readback_pv.get()
        if args.verbose:
            print('Setting motor to calibrated position: {}'
                  ''.format(calibrated_position))
        cam.calibrate_motor(calibrated_position, verbose=args.verbose)

    if args.compare_to:
        plt.figure(10)
        label1 = args.load if args.load else 'Calibrated'
        compare_fits(
            data,
            labels=[label1, args.compare_to],
            fit_info_dicts=[data['calibration'], fit_results],
            line=args.line,
        )
        plt.figure(0)

    if args.plot or args.compare_to:
        plt.ioff()
        plt.tight_layout()
        print('(See plot)', file=sys.stderr)
        plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    commands = parser.add_mutually_exclusive_group(required=True)
    commands.add_argument('--calibrate', type=str,
                          help='Calibrate motors with the given EPICS prefix')
    commands.add_argument('--load', type=str,
                          help='Load calibration data from file')

    parser.add_argument('--store-to-pv',
                        action='store_true',
                        help='Store calibration data on motor')
    parser.add_argument('--save-to', type=str,
                        help='Save calibration data to file')

    parser.add_argument('--serial', '-s', type=str,
                        help='Specify a relevant serial number')
    parser.add_argument('--line', '-l', type=str, choices=('hxr', 'sxr'),
                        default='hxr', required=True,
                        help='Specify the undulator line')
    parser.add_argument('--number', '-n', type=int,
                        help='Specify the cam positioner number')
    parser.add_argument('--plot', '-p',
                        action='store_true',
                        help='Plot relevant calibration information')
    parser.add_argument('--velocity', type=float, default=1.0,
                        help='Velocity for calibration')
    parser.add_argument('--compare-to', type=str,
                        help='Compare calibration results with this file')
    parser.add_argument('--dwell',
                        type=float, default=1.0,
                        help='Dwell time after move')
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='Verbose operations')
    args = parser.parse_args()
    main(args)
