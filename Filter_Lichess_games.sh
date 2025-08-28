#Shell script Used in conjunction with blitz_tags to filter datasets to only return blitz games

MONTH_YEAR="2025-06"

INPUT_PGN_FILE="lichess_db_standard_rated_${MONTH_YEAR}.pgn" 

OUTPUT_BLITZ_PGN="El_New_Blitz.pgn" 

BLITZ_TAGS_FILE="blitz.tags" 


echo "Starting full extraction of games"

