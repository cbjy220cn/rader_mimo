%% Script to Absolutely Verify the Rotation Logic of ArrayPlatform
% This script performs a simple, clear-cut test to confirm that the
% ArrayPlatform class correctly calculates the positions of its elements
% when a rotational trajectory is applied.

clear; clc; close all;

fprintf('--- Starting ArrayPlatform Rotation Verification ---\n\n');

% 1. Define a simple 2-element array along the X-axis
physical_elements = [0, 0, 0;  % Element 1 at origin
                     1, 0, 0]; % Element 2 at x=1

array = ArrayPlatform(physical_elements, 1, 1:2);

% 2. Get initial positions (at t=0)
pos_t0 = array.get_physical_positions(0);
fprintf('Initial positions (t=0):\n');
disp(pos_t0);

fprintf('Expected initial positions:\n');
disp([0 0 0; 1 0 0]);
assert(all(all(abs(pos_t0 - [0 0 0; 1 0 0]) < 1e-9)), 'Initial position is incorrect!');
fprintf('--> Initial position test PASSED.\n\n');


% 3. Define a 90-degree rotation trajectory around the Z-axis over 1 second
rotation_angle_deg = 90;
total_time = 1;
omega_dps = rotation_angle_deg / total_time; % 90 deg/sec

rotation_trajectory = @(t) struct('position', [0,0,0], 'orientation', [0, 0, omega_dps * t]);

% Apply the trajectory. Since ArrayPlatform is a handle class, this modifies
% the object 'array' directly.
array.set_trajectory(rotation_trajectory);

% 4. Get final positions (at t=1)
pos_t1 = array.get_physical_positions(1);
fprintf('Final positions after 90-deg rotation (t=1):\n');
disp(pos_t1);

% 5. Compare with the expected result
expected_pos_t1 = [0, 0, 0;  % Origin doesn't move
                   0, 1, 0]; % [1,0,0] rotated by +90 deg around Z becomes [0,1,0]
fprintf('Expected final positions:\n');
disp(expected_pos_t1);

error = abs(pos_t1 - expected_pos_t1);
assert(all(all(error < 1e-9)), 'Final rotated position is incorrect!');
fprintf('--> Final rotated position test PASSED.\n\n');

fprintf('--- VERIFICATION SUCCESSFUL: ArrayPlatform rotation logic is correct. ---\n');
