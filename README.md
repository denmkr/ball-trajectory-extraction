# Extraction of Tennis Ball Trajectories from Video Recordings

### Ball Trajectory Extraction (one tennis ball presented in video)
To run a script and get a file with ball positions for each frame, you need run the following command:\
`python3 extract_ball_data_one_ball.py --input_video_path=path-to-video --output_file_path=path-to-extracted-csv-file.csv`

Example:\
`python3 extract_ball_data_one_ball.py --input_video_path=videos/test2.mp4 --output_file_path=output/one_ball_extracted_csv_file.csv`

### Ball Trajectory Extraction (multiple tennis balls presented in video)
To run a script and get a file with ball positions for each frame (multiple games and balls captured in video), you need run the following command:\
`python3 extract_ball_data_multiple_balls.py --input_video_path=path-to-video --output_file_path=path-to-extracted-csv-file.csv --court_coordinates='[[left_top_x, left_top_y], [right_top_x, right_top_y], [right_bottom_x, right_bottom_y], [left_bottom_x, left_bottom_y]]'`

Example:\
`python3 extract_ball_data_multiple_balls.py --input_video_path=videos/test2.mp4 --output_file_path=output/multiple_balls_extracted_csv_file.csv --court_coordinates='[[153.6, 334.37722419928826], [687.36, 295.94306049822063], [1560.96, 545.7651245551601], [385.912, 814.8042704626334]]'`

This script analyzes multiple ball positions and selects correct one based on the court boundaries. The values for 'court_coordinates' can be obtained using game annotation tool. After uploading a video, click the button "Calibrate" and specify the positions on the court which are highlighted in top right court scheme panel. After clicking the button "Complete", the coordinates appear above the court scheme.

### Key Frames Extraction
To extract key events file (with ball bounces and player hits frames), run the command:\
`python3 extract_key_frames.py --input_file_path='path-to-raw-ball-positions-file.csv' --output_file_path='path-to-extracted-csv-file.csv'`

Example:\
`python3 extract_key_frames.py --input_file_path='ball_raw_data.csv' --output_file_path='output/key_frame_data.csv'`

You can use both these output files (after running ball trajectory extraction and key frames extraction scripts) in game annotation tool. These files have the same csv structure and therefore can be processed by the tool similarly. Just upload a csv file in "Ball data csv" button after uploading a video.