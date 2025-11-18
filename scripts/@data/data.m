classdef data
%DATA A class for managing experiment data logging.
%   Creates a unique, timestamped directory for each experiment run
%   and provides methods to save figures and data.

    properties
        output_dir % The unique directory for this run's output
    end
    
    methods
        function obj = data()
            % Constructor: Creates a unique directory for this experiment run.
            base_dir = 'output';
            if ~exist(base_dir, 'dir')
                mkdir(base_dir);
            end
            
            timestamp = datestr(now, 'yyyymmdd_HHMMSS');
            obj.output_dir = fullfile(base_dir, timestamp);
            mkdir(obj.output_dir);
            
            fprintf('Results for this run will be saved in: %s\n', obj.output_dir);
        end
        
        function save_figure(obj, fig_handle, filename)
            %SAVE_FIGURE Saves a figure to both .fig and .png formats.
            if ~isgraphics(fig_handle, 'figure')
                warning('Invalid figure handle provided. Skipping save.');
                return;
            end
            
            % Create full file paths without extension
            base_filepath = fullfile(obj.output_dir, filename);
            
            % Save as .fig for future editing
            savefig(fig_handle, [base_filepath, '.fig']);
            
            % Save as .png for easy viewing
            exportgraphics(fig_handle, [base_filepath, '.png'], 'Resolution', 300);
            
            fprintf('Figure saved to %s.fig and %s.png\n', base_filepath, base_filepath);
        end
        
        function save_data(obj, filename, data_to_save)
            %SAVE_DATA Saves workspace variables to a .mat file.
            %   data_to_save should be a struct. The fields of the struct
            %   will be saved as variables in the .mat file.
            if ~isstruct(data_to_save)
                error('Data to be saved must be provided as a struct.');
            end
            
            full_filepath = fullfile(obj.output_dir, [filename, '.mat']);
            
            % Use save command with the -struct flag
            save(full_filepath, '-struct', 'data_to_save');
            
            fprintf('Data saved to %s\n', full_filepath);
        end
    end
end
