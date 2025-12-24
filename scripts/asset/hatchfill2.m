function H = hatchfill2(A, varargin)
% HATCHFILL2 - Fill a patch with hatching patterns
%
% H = HATCHFILL2(A) fills the patch A with hatching
% H = HATCHFILL2(A, STYL) uses style STYL for hatching
% H = HATCHFILL2(A, STYL, ANG) uses angle ANG for the hatch lines
% H = HATCHFILL2(A, STYL, ANG, SPACING) uses spacing SPACING
% H = HATCHFILL2(..., 'PropertyName', PropertyValue, ...) sets properties
%
% STYL: 'single' - single lines (default)
%       'cross'  - crosshatch
%       'fill'   - solid fill
%       'none'   - no fill
%
% Default angle is 45 degrees
% Default spacing is 5
%
% Properties:
%   'HatchAngle'     - Angle of hatch lines (degrees)
%   'HatchDensity'   - Density of hatch lines (lines per unit)
%   'HatchColor'     - Color of hatch lines
%   'HatchLineWidth' - Width of hatch lines
%   'HatchLineStyle' - Style of hatch lines ('-', '--', ':', '-.')
%
% Based on hatchfill by Neil Tandon and hatchfill2 by Takeshi Ikuma

% Parse inputs
p = inputParser;
p.addRequired('A');
p.addOptional('Style', 'single', @ischar);
p.addOptional('Angle', 45, @isnumeric);
p.addOptional('Spacing', 5, @isnumeric);
p.addParameter('HatchAngle', [], @isnumeric);
p.addParameter('HatchDensity', 40, @isnumeric);
p.addParameter('HatchColor', 'k');
p.addParameter('HatchLineWidth', 0.5, @isnumeric);
p.addParameter('HatchLineStyle', '-', @ischar);
p.addParameter('HatchSpacing', [], @isnumeric);

p.parse(A, varargin{:});

style = lower(p.Results.Style);
angle = p.Results.Angle;
if ~isempty(p.Results.HatchAngle)
    angle = p.Results.HatchAngle;
end
density = p.Results.HatchDensity;
hatchColor = p.Results.HatchColor;
lineWidth = p.Results.HatchLineWidth;
lineStyle = p.Results.HatchLineStyle;

if ~isempty(p.Results.HatchSpacing)
    spacing = p.Results.HatchSpacing;
else
    spacing = 100 / density;
end

if strcmp(style, 'none') || strcmp(style, 'fill')
    H = [];
    return;
end

% Handle bar chart objects
if isa(A, 'matlab.graphics.chart.primitive.Bar')
    H = hatchfill_bar(A, style, angle, spacing, hatchColor, lineWidth, lineStyle);
    return;
end

% Handle patch objects
if ~isa(A, 'matlab.graphics.primitive.Patch')
    error('Input must be a patch or bar object');
end

H = hatchfill_patch(A, style, angle, spacing, hatchColor, lineWidth, lineStyle);

end

function H = hatchfill_bar(barObj, style, angle, spacing, hatchColor, lineWidth, lineStyle)
% Fill bar chart with hatching

H = [];
ax = barObj.Parent;
hold_state = ishold(ax);
hold(ax, 'on');

% Get bar data
xData = barObj.XEndPoints;
yData = barObj.YData;
barWidth = barObj.BarWidth;

% Calculate actual bar width based on number of bars in group
nBars = 1;
siblings = ax.Children;
for i = 1:length(siblings)
    if isa(siblings(i), 'matlab.graphics.chart.primitive.Bar')
        nBars = nBars + 1;
    end
end
nBars = nBars - 1;
if nBars < 1, nBars = 1; end

actualWidth = barWidth * 0.8 / nBars;

for i = 1:length(xData)
    x = xData(i);
    y = yData(i);
    
    if y <= 0
        continue;
    end
    
    % Create vertices for bar rectangle
    x_left = x - actualWidth/2;
    x_right = x + actualWidth/2;
    
    vertices = [x_left, 0; x_right, 0; x_right, y; x_left, y];
    
    % Generate hatch lines
    h = draw_hatch_in_polygon(ax, vertices, style, angle, spacing, hatchColor, lineWidth, lineStyle);
    H = [H; h];
end

if ~hold_state
    hold(ax, 'off');
end

end

function H = hatchfill_patch(patchObj, style, angle, spacing, hatchColor, lineWidth, lineStyle)
% Fill patch with hatching

H = [];
ax = patchObj.Parent;
hold_state = ishold(ax);
hold(ax, 'on');

% Get patch vertices
xData = patchObj.XData;
yData = patchObj.YData;

if iscell(xData)
    % Multiple patches
    for k = 1:length(xData)
        vertices = [xData{k}(:), yData{k}(:)];
        vertices = vertices(~isnan(vertices(:,1)), :);
        h = draw_hatch_in_polygon(ax, vertices, style, angle, spacing, hatchColor, lineWidth, lineStyle);
        H = [H; h];
    end
else
    vertices = [xData(:), yData(:)];
    vertices = vertices(~isnan(vertices(:,1)), :);
    h = draw_hatch_in_polygon(ax, vertices, style, angle, spacing, hatchColor, lineWidth, lineStyle);
    H = [H; h];
end

if ~hold_state
    hold(ax, 'off');
end

end

function H = draw_hatch_in_polygon(ax, vertices, style, angle, spacing, hatchColor, lineWidth, lineStyle)
% Draw hatch lines inside a polygon

H = [];

if size(vertices, 1) < 3
    return;
end

% Get bounding box
xmin = min(vertices(:,1));
xmax = max(vertices(:,1));
ymin = min(vertices(:,2));
ymax = max(vertices(:,2));

width = xmax - xmin;
height = ymax - ymin;
diagonal = sqrt(width^2 + height^2);

% Center of bounding box
cx = (xmin + xmax) / 2;
cy = (ymin + ymax) / 2;

% Generate hatch lines
angle_rad = angle * pi / 180;
cos_a = cos(angle_rad);
sin_a = sin(angle_rad);

% Number of lines
nLines = ceil(diagonal / spacing) + 2;

% Generate lines perpendicular to angle direction
for i = -nLines:nLines
    offset = i * spacing;
    
    % Line endpoints (extended beyond bounding box)
    if abs(cos_a) > abs(sin_a)
        % More horizontal
        x1 = cx - diagonal;
        x2 = cx + diagonal;
        y1 = cy + offset/cos_a - (x1 - cx) * tan(angle_rad);
        y2 = cy + offset/cos_a - (x2 - cx) * tan(angle_rad);
    else
        % More vertical
        y1 = cy - diagonal;
        y2 = cy + diagonal;
        x1 = cx + offset/sin_a - (y1 - cy) / tan(angle_rad);
        x2 = cx + offset/sin_a - (y2 - cy) / tan(angle_rad);
    end
    
    % Clip line to polygon
    [xi, yi] = clip_line_to_polygon([x1, x2], [y1, y2], vertices);
    
    if ~isempty(xi)
        h = plot(ax, xi, yi, 'Color', hatchColor, 'LineWidth', lineWidth, ...
            'LineStyle', lineStyle, 'Clipping', 'on');
        H = [H; h];
    end
end

% Cross hatch
if strcmp(style, 'cross')
    h2 = draw_hatch_in_polygon(ax, vertices, 'single', angle + 90, spacing, hatchColor, lineWidth, lineStyle);
    H = [H; h2];
end

end

function [xi, yi] = clip_line_to_polygon(xline, yline, vertices)
% Clip a line segment to a polygon using simple intersection

xi = [];
yi = [];

n = size(vertices, 1);
if n < 3
    return;
end

% Find all intersections with polygon edges
intersections = [];

x1 = xline(1); y1 = yline(1);
x2 = xline(2); y2 = yline(2);

for i = 1:n
    j = mod(i, n) + 1;
    
    x3 = vertices(i, 1); y3 = vertices(i, 2);
    x4 = vertices(j, 1); y4 = vertices(j, 2);
    
    % Line-line intersection
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4);
    
    if abs(denom) < 1e-10
        continue;
    end
    
    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom;
    u = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / denom;
    
    if u >= 0 && u <= 1
        px = x1 + t*(x2-x1);
        py = y1 + t*(y2-y1);
        intersections = [intersections; px, py, t];
    end
end

if size(intersections, 1) < 2
    return;
end

% Sort by parameter t
intersections = sortrows(intersections, 3);

% Take pairs of intersections (entry/exit points)
for i = 1:2:size(intersections, 1)-1
    xi = [xi, intersections(i, 1), intersections(i+1, 1), NaN];
    yi = [yi, intersections(i, 2), intersections(i+1, 2), NaN];
end

% Remove trailing NaN
if ~isempty(xi) && isnan(xi(end))
    xi = xi(1:end-1);
    yi = yi(1:end-1);
end

end

