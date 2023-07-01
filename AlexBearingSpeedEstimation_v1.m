% This script extracts the rotational speed of the following bearing's components:
%	- shaft
%	- train
%	- outer race
% and saves them in a Matlab data file and a .txt file.
% 
% Author: Toby Verwimp
% Last update: 16/06/2023

close all
clear all
clc


%% LOAD VIDEO AND DATA
videoFilePath = fullfile(fileparts(pwd),"\rpm_variable.avi");
video = VideoReader(videoFilePath);
[videoFolder,videoName,~] = fileparts(videoFilePath);
dataFilePath = strcat(fullfile(videoFolder,strcat('Data_',videoName)),'.mat');


%% INITIALIZE VARIABLES & PARAMETERS
	%% Variables
	speedShaft = zeros(video.NumFrames,1); % Speed of shaft [rps, Hz]
	angleShaft = zeros(video.NumFrames,1); % Angle of shaft [rad]
	speedTrain = zeros(video.NumFrames,1); % Speed of train/cage [rps, Hz]
	angleTrain = zeros(video.NumFrames,1); % Angle of train/cage [rad]
	speedOuter = zeros(video.NumFrames,1); % Speed of outer race [rps, Hz]
	angleOuter = zeros(video.NumFrames,1); % Angle of outer race [rad]

	%% Parameters
	frameRate = 2e3; % Frame rate of video [fps, Hz]
	speedResolution = 1/60; % Desired speed resolution [rps, Hz]
	maxSpeed = 50; % Expected maximum speed [rps, Hz]
	encoderResolution = frameRate/speedResolution; % Needed encoder resolution to achieve the speed resolution [ppr]
	angleResolution = 2*pi/encoderResolution; % Needed angle resolution to achieve the speed resolution [rad]
	maxAngle = maxSpeed/frameRate*2*pi; % Maximum angle over which can be rotated after 1 frame period [rad]
	nAnglesCorrelation = 2*maxAngle/angleResolution + 1; % Maximum number of angles to correlate over [-]
	angleDiffs = -maxAngle:angleResolution:maxAngle; % Array with angles to correlate over [rad]


%% LOOK FOR VIDEO FRAMES THAT ARE INVALID
%%%% UNCOMMENT TO DO THIS AGAIN (TAKES TIME)
% invalidFramesTotal = [];
% for iFrame = 1:video.NumFrames
% 	videoFrame = read(video,iFrame);
% 	if any(~any(videoFrame,2))
% 		invalidFramesTotal(end+1) = iFrame;
% 	end
% end
% frameNumbers = 1:video.NumFrames;
% frameNumbers = frameNumbers(~ismember(frameNumbers,invalidFramesTotal));
% invalidFrames = struct('invalidFramesTotal',invalidFramesTotal);
% if exist(dataFilePath,"file")
% 	save(dataFilePath,"invalidFrames","frameNumbers","-append");
% else
% 	save(dataFilePath,"invalidFrames","frameNumbers");
% end
data = load(dataFilePath);
frameNumbers = data.frameNumbers;
invalidFrames = data.invalidFrames;


%% CIRCLES AND SHAFT'S CENTER EXTRACTION
%%%% UNCOMMENT TO DO THIS AGAIN (TAKES TIME)
% CirclesExtraction(video,dataFilePath);


%% LOOK FOR VIDEO FRAMES THAT ARE INVALID PER REGION OF INTEREST (ROI)
invalidFramesShaftROI = [];
invalidFramesTrainROI = [];
invalidFramesOuterRaceROI = [];
centerAvg = mean(data.circles.centers,1);
	
for iFrame = invalidFrames.invalidFramesTotal
	videoFrame = read(video,iFrame);

	%% Get rows that only contain zeros
	zeroRows = [];
	for iRow = 1:video.Height
		if all(videoFrame(iRow,:) == 0)
			zeroRows(end+1) = iRow;
		end
	end

	%% Check for each ROI whether it contains a row with only zeros
	if any(zeroRows>=centerAvg(1)-data.circles.radiusShaftOuter & zeroRows<=centerAvg(1)+data.circles.radiusShaftOuter) % In shaft ROI
		invalidFramesShaftROI(end+1) = iFrame;
		invalidFramesTrainROI(end+1) = iFrame;
	elseif any(zeroRows>=centerAvg(1)-data.circles.radiusOuterRaceInner & zeroRows<=centerAvg(1)+data.circles.radiusOuterRaceInner) % In train/cage ROI
		invalidFramesTrainROI(end+1) = iFrame;
	end
	if any(zeroRows>=min([data.circles.positionLeftRectangle(2),data.circles.positionRightRectangle(2)]) & ...
			zeroRows<=max([data.circles.positionLeftRectangle(2)+data.circles.positionLeftRectangle(4),data.circles.positionRightRectangle(2)+data.circles.positionRightRectangle(4)])) % In outer race ROI
		invalidFramesOuterRaceROI(end+1) = iFrame;
	end
end
invalidFrames.("invalidFramesShaftROI") = invalidFramesShaftROI;
invalidFrames.("invalidFramesTrainROI") = invalidFramesTrainROI;
invalidFrames.("invalidFramesOuterRaceROI") = invalidFramesOuterRaceROI;
save(dataFilePath,"invalidFrames","-append");


%% SPEED EXTRACTION --> UNDER CONSTRUCTION
	%% Get pixel coordinates of all ROIs
	centerAvg = mean(data.circles.centers,1);
	pixelCoordinatesShaft = [];
	pixelCoordinatesTrain = [];
	pixelCoordinatesOuter = [];
	posLR = data.circles.positionLeftRectangle;
	posRR = data.circles.positionRightRectangle;
	for y = 1:video.Height
    	for x = 1:video.Width
			pixelIdx = (y-1)*video.Width + x;
			distance = sqrt((x-centerAvg(2))^2 + (y-centerAvg(1))^2); % Calculate the distance between the current pixel and the circle center
			if (distance<=data.circles.radiusShaftOuter) && (distance>=data.circles.radiusShaftInner) % Check if the pixel is within the shaft's ROI
				pixelCoordinatesShaft(end+1) = pixelIdx; % Add the pixel coordinates to the array
			elseif (distance<=data.circles.radiusOuterRaceInner) && (distance>=data.circles.radiusInnerRace) % Check if the pixel is within the train's ROI
				pixelCoordinatesTrain(end+1) = pixelIdx; % Add the pixel coordinates to the array
			elseif (x>=posLR(1) && x<=posLR(1)+posLR(3) && y>=posLR(2) && y<=posLR(2)+posLR(4)) || ...
					(x>=posRR(1) && x<=posRR(1)+posRR(3) && y>=posRR(2) && y<=posRR(2)+posRR(4)) % Check if the pixel is within the outer race's ROI
				pixelCoordinatesOuter(end+1) = pixelIdx; % Add the pixel coordinates to the array
			end
    	end
	end

	%% Iterate over all frames
	frame = read(video,1);
	for iFrame = 2:video.NumFrames
		refFrame = frame;
		if ismember(iFrame,[invalidFrames.invalidFramesShaftROI,invalidFrames.invalidFramesTrainROI,invalidFrames.invalidFramesOuterRaceROI])
			continue
		end
		frame = read(video,iFrame); % Read frame
        tic
        kk=0;
%         figure(1);
        for Radii=140:170
            kk=kk+1;
            centerX=data.circles.centers(iFrame,1);
            centerY=data.circles.centers(iFrame,2);
            angleResolution=2*pi/(2^10);
            radius=Radii;
            [oneCircle, xInt, yInt, xUnit, yUnit] = ExtractRadiusPoints(frame, centerX, centerY, radius, angleResolution);
            InnerVec(kk,:)=oneCircle;
%             InnerX(kk,:)=xUnit;
%             InnerY(kk,:)=yUnit;
%             figure(1); imagesc(frame), hold on, 
%             colormap((gray)), axis equal
%             scatter(xUnit,yUnit,'r','filled'), hold off
%             pause(0.1)
        end
        InnerFull(:,iFrame-1)=sum(InnerVec);

        kk=0;
        for Radii=210:280
            kk=kk+1;
            centerX=data.circles.centers(iFrame,1);
            centerY=data.circles.centers(iFrame,2);
            angleResolution=2*pi/(2^12);
            radius=Radii;
            [oneCircle, xInt, yInt, xUnit, yUnit] = ExtractRadiusPoints(frame, centerX, centerY, radius, angleResolution);
            CageVec(kk,:)=oneCircle;
            InnerX(kk,:)=xUnit;
            InnerY(kk,:)=yUnit;
            figure(1); imagesc(frame), hold on, axis equal
            scatter(xUnit,yUnit,'r','filled'), hold off
            pause(0.1)
        end
        CageFull(:,iFrame-1)=sum(CageVec);


        kk=0;
        for Radii=320:370
            kk=kk+1;
            centerX=data.circles.centers(iFrame,1);
            centerY=data.circles.centers(iFrame,2);
            angleResolution=2*pi/(2^13);
            radius=Radii;
            [oneCircle, xInt, yInt, xUnit, yUnit] = ExtractRadiusPoints(frame, centerX, centerY, radius, angleResolution);
            OuterVec(kk,:)=oneCircle;
%             InnerX(kk,:)=xUnit;
%             InnerY(kk,:)=yUnit;
%             figure(1); imagesc(frame), hold on, axis equal
%             scatter(xUnit,yUnit,'r','filled'), hold off
%             pause(0.1)
        end
        OuterFull(:,iFrame-1)=sum(OuterVec);
        toc

% 		correlationShaft = zeros(nAnglesCorrelation,1); % Correlation of ROI with rotated ROI as function of the angle
% 		correlationTrain = zeros(nAnglesCorrelation,1);
% 		correlationOuter = zeros(nAnglesCorrelation,1);
% 		for iAngle = 1:nAnglesCorrelation % Iterate over rotations
% 			frameRotated = rotateAround(frame,centerAvg(1),centerAvg(2),angleDiffs(iAngle)*180/pi,'bicubic');
% 			correlationShaft(iAngle) = sum(frame(pixelCoordinatesShaft).*refFrame(pixelCoordinatesShaft));
% 			correlationTrain(iAngle) = sum(frame(pixelCoordinatesTrain).*refFrame(pixelCoordinatesTrain));
% 			correlationOuter(iAngle) = sum(frame(pixelCoordinatesOuter).*refFrame(pixelCoordinatesOuter));
% 		end
% 		[~,maxIdxShaft] = max(correlationShaft);
% 		[~,maxIdxTrain] = max(correlationTrain);
% 		[~,maxIdxOuter] = max(correlationOuter);
% 		angleShaft(iFrame) = angleShaft(iFrame-1) + angleDiffs(maxIdxShaft);
% 		angleTrain(iFrame) = angleTrain(iFrame-1) + angleDiffs(maxIdxTrain);
% 		angleOuter(iFrame) = angleOuter(iFrame-1) + angleDiffs(maxIdxOuter);
% 		speedShaft(iFrame) = angleDiffs(maxIdxShaft)*frameRate/2/pi;
% 		speedTrain(iFrame) = angleDiffs(maxIdxTrain)*frameRate/2/pi;
% 		speedOuter(iFrame) = angleDiffs(maxIdxOuter)*frameRate/2/pi;
	end



speedShaft(1) = speedShaft(2);
speedTrain(1) = speedTrain(2);
speedOuter(1) = speedOuter(2);

% TODO: linearly interpolate the speed for invalid frames