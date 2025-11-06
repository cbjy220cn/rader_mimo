classdef Target
    % Target Class: A simple container for target properties.
    
    properties
        initial_position % [x, y, z] initial position in meters
        velocity         % [vx, vy, vz] velocity in m/s
        rcs              % Radar Cross Section in m^2
    end
    
    methods
        function obj = Target(initial_position, velocity, rcs)
            % Constructor to initialize a target object.
            if nargin > 0
                obj.initial_position = initial_position;
                obj.velocity = velocity;
                obj.rcs = rcs;
            end
        end
        
        function position = get_position_at(obj, t)
            % Calculates the target's position at a given time t.
            position = obj.initial_position + obj.velocity * t;
        end
    end
end
