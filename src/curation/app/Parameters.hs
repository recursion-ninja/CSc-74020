{-# Language DerivingStrategies #-}
{-# Language GeneralizedNewtypeDeriving #-}
{-# Language LambdaCase #-}
{-# Language OverloadedStrings #-}

module Parameters
  ( -- * Parser
    parseCommandLineOptions
    -- * Data-type
  , Parameters()
    -- * Accessors
  , filesWithRanks
  , filesWithStats
  , outputFile
  ) where

import Data.Foldable
import Data.List.NonEmpty (NonEmpty(..))
import Options.Applicative


data  Parameters
    = Parameters
    { filesWithRanks' :: [FilePath]
    , filesWithStats' :: [FilePath]
    , outputFile'     ::  FilePath
    }
    deriving stock (Show)


-- |
-- Command to parse the command line options.
parseCommandLineOptions :: IO Parameters
parseCommandLineOptions = customExecParser preferences parserInformation
  where
    preferences = prefs $ fold [showHelpOnError, showHelpOnEmpty]


-- |
-- Information regarding which command line options are valid and how they are
-- parsed and interpreted.
parserInformation :: ParserInfo Parameters
parserInformation = info commandLineOptions fullDesc
  where
    commandLineOptions =
        Parameters
          <$> inputRanksSpec
          <*> inputStatsSpec
          <*> outputFileSpec

    inputRanksSpec :: Parser [FilePath]
    inputRanksSpec = some . strOption $ fold
        [ short 'r'
        , long  "ranks"
        , help  "ranking file(s)"
        , metavar "[ FILE ]"
        ]

    inputStatsSpec :: Parser [FilePath]
    inputStatsSpec = some . strOption $ fold
        [ short 's'
        , long  "stats"
        , help  "stat-block file(s)"
        , metavar "[ FILE ]"
        ]

    outputFileSpec :: Parser FilePath
    outputFileSpec = strOption $ fold
        [ short 'o'
        , long  "output"
        , value defaultOutputCSV
        , help  $ fold ["Output file", " (default ", defaultOutputCSV, ")"]
        , metavar "FILE"
        ]


-- |
-- Accessor for monster ranking 'FilePath' values.
filesWithRanks :: Parameters -> [FilePath]
filesWithRanks = fmap unquote . filesWithRanks'


-- |
-- Accessor for monster statistic 'FilePath' values.
filesWithStats :: Parameters -> [FilePath]
filesWithStats = fmap unquote . filesWithStats'


-- |
-- Accessor for the output CSV 'FilePath' value.
outputFile :: Parameters -> FilePath
outputFile = unquote . outputFile'


defaultOutputCSV :: FilePath
defaultOutputCSV = "dnd-5e-monsters.csv"


unquote :: FilePath -> FilePath
unquote input =
    let quotes = "\"'" :: String
    in  case input of
            q:x:xs | q `elem` quotes -> case unsnoc $ x:|xs of
                (c,cs) | q == c -> cs
                _ -> input
            _ -> input


unsnoc :: NonEmpty a -> (a, [a])
unsnoc (x:|xs) =
    case xs of
      []   -> (x, [])
      y:ys -> (x:) <$> unsnoc (y:|ys)
