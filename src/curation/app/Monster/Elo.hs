{-# Language DeriveAnyClass #-}
{-# Language DeriveGeneric #-}
{-# Language DerivingStrategies #-}
{-# Language GeneralizedNewtypeDeriving #-}
{-# Language OverloadedLists #-}
{-# Language OverloadedStrings #-}
{-# Language Strict #-}

{-# Language LambdaCase #-}
{-# Language RecordWildCards #-}

module Monster.Elo
  ( EloReference()
  , Label()
  , extractionKeyForRanks
  , toRankingPair
  ) where

import Control.Applicative
import Data.Aeson          hiding ((.=))
import Data.Aeson.Types    (Parser)
import Data.Csv            (DefaultOrdered(..), ToNamedRecord(..), namedRecord, (.=))
import Data.Maybe
import Data.String
import Data.Text           (Text, unpack)
import GHC.Generics
import Monster.Label
import Prelude             hiding (break, lookup, null)
import Text.HTML.TagSoup
import Text.Read


extractionKeyForRanks :: Key
extractionKeyForRanks = "data"


toRankingPair :: EloReference -> (Label, Int)
toRankingPair = (,) <$> Label . nameForElo <*> recoredElo


data  EloReference
    = EloReference
    { nameForElo :: String
    , recoredElo :: Int
    }
    deriving stock    (Eq, Generic, Ord, Show)


instance FromJSON EloReference where

    parseJSON = withObject "Item" $ \obj ->
        EloReference <$> parseName obj <*> parseEloRank obj


instance ToNamedRecord EloReference where

    toNamedRecord EloReference{..} = namedRecord
        [ "Name"     .= nameForElo
        , "Elo Rank" .= recoredElo
        ]


instance DefaultOrdered EloReference where

    headerOrder = const ["Name", "Elo Rank"]


parseName :: Object -> Parser String
parseName obj = obj .:? "f2" >>= getTextHTML "UNNAMED!?"


parseEloRank :: Object -> Parser Int
parseEloRank obj = obj .:? "f5" >>= getFromHTML 0


getTextHTML :: String -> Maybe Text -> Parser String
getTextHTML def may = pure . fromMaybe def $ may >>= tryGetTextOfHTML


getFromHTML :: Read a => a -> Maybe Text -> Parser a
getFromHTML def may = pure . fromMaybe def $ may >>= tryGetTextOfHTML >>= readMaybe


tryGetTextOfHTML :: Text -> Maybe String
tryGetTextOfHTML = getTextFromTags . unpack
  where
    getTextFromTags :: String -> Maybe String
    getTextFromTags = tryHead . foldMap getTextTag . parseTags

    tryHead :: [a] -> Maybe a
    tryHead =
      \case
        []  -> Nothing
        x:_ -> Just x

    getTextTag :: Tag String -> [String]
    getTextTag =
      \case
        TagText v -> [v]
        _         -> []
