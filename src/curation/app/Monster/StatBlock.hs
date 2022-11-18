{-# Language DeriveAnyClass #-}
{-# Language DeriveGeneric #-}
{-# Language DerivingStrategies #-}
{-# Language GeneralizedNewtypeDeriving #-}
{-# Language LambdaCase #-}
{-# Language OverloadedStrings #-}
{-# Language RecordWildCards #-}
{-# Language Strict #-}

module Monster.StatBlock
  ( StatBlock()
  , ModifierBlock()
  , Label()
  , extractionKeyForStats
  , getMonsterName
  , setEloRank
  ) where

import Control.Applicative
import Data.Aeson          hiding ((.=))
import Data.Aeson.Types    (Parser)
import Data.Char
import Data.Csv            (DefaultOrdered(..), Field, ToField(..), ToNamedRecord(..), namedRecord, (.=))
import Data.Foldable       (fold, null, toList)
import Data.Functor        ((<&>))
import Data.List           (intercalate, intersperse)
import Data.Maybe
import Data.Scientific
import Data.String
import Data.Text           (Text, uncons, unpack)
import Data.Vector qualified as V
import GHC.Exts            (IsList(fromList))
import GHC.Generics
import Monster.Label
import Prelude             hiding (break, lookup, null)
import Text.Read           (readMaybe)


extractionKeyForStats :: Key
extractionKeyForStats = "monster"


newtype Bonus = Bonus Int
    deriving newtype (Enum, Eq, Num, Ord, Show)
    deriving stock   (Generic)


data  MonsterSize
    = Tiny
    | Small
    | Medium
    | Large
    | Huge
    | Giant
    deriving stock    (Bounded, Enum, Eq, Generic, Ord, Show)


data  StatBlock
    = StatBlock
    { monsterName                          :: String
    , monsterType                          :: String
    , monsterSize                          :: MonsterSize
    , monsterArmor                         :: Word
    , monsterHitPoints                     :: Word

    , monsterSpeedBurrow                   :: Word
    , monsterSpeedClimb                    :: Word
    , monsterSpeedFly                      :: Word
    , monsterSpeedSwim                     :: Word
    , monsterSpeedWalk                     :: Word

    , monsterStatStrength                  :: Word
    , monsterStatDexterity                 :: Word
    , monsterStatConstitution              :: Word
    , monsterStatIntelligence              :: Word
    , monsterStatWisdom                    :: Word
    , monsterStatCharisma                  :: Word

    , monsterSaveStrength                  :: Bonus
    , monsterSaveDexterity                 :: Bonus
    , monsterSaveConstitution              :: Bonus
    , monsterSaveIntelligence              :: Bonus
    , monsterSaveWisdom                    :: Bonus
    , monsterSaveCharisma                  :: Bonus

    , monsterBlindSight                    :: Bool
    , monsterDarkVision                    :: Bool
    , monsterTremorsense                   :: Bool
    , monsterTrueSight                     :: Bool

    , monsterDamageImmuneAcid              :: Bool
    , monsterDamageImmuneBludgeoning       :: Bool
    , monsterDamageImmuneCold              :: Bool
    , monsterDamageImmuneFire              :: Bool
    , monsterDamageImmuneForce             :: Bool
    , monsterDamageImmuneLightning         :: Bool
    , monsterDamageImmuneNecrotic          :: Bool
    , monsterDamageImmunePiercing          :: Bool
    , monsterDamageImmunePoison            :: Bool
    , monsterDamageImmunePsychic           :: Bool
    , monsterDamageImmuneRadiant           :: Bool
    , monsterDamageImmuneSlashing          :: Bool
    , monsterDamageImmuneThunder           :: Bool

    , monsterDamageResistAcid              :: Bool
    , monsterDamageResistBludgeoning       :: Bool
    , monsterDamageResistCold              :: Bool
    , monsterDamageResistFire              :: Bool
    , monsterDamageResistForce             :: Bool
    , monsterDamageResistLightning         :: Bool
    , monsterDamageResistNecrotic          :: Bool
    , monsterDamageResistPiercing          :: Bool
    , monsterDamageResistPoison            :: Bool
    , monsterDamageResistPsychic           :: Bool
    , monsterDamageResistRadiant           :: Bool
    , monsterDamageResistSlashing          :: Bool
    , monsterDamageResistThunder           :: Bool

    , monsterInflictConditionBlinded       :: Bool
    , monsterInflictConditionCharmed       :: Bool
    , monsterInflictConditionDeafened      :: Bool
    , monsterInflictConditionFrightened    :: Bool
    , monsterInflictConditionGrappled      :: Bool
    , monsterInflictConditionIncapacitated :: Bool
    , monsterInflictConditionInvisible     :: Bool
    , monsterInflictConditionParalyzed     :: Bool
    , monsterInflictConditionPetrified     :: Bool
    , monsterInflictConditionPoisoned      :: Bool
    , monsterInflictConditionProne         :: Bool
    , monsterInflictConditionRestrained    :: Bool
    , monsterInflictConditionStunned       :: Bool
    , monsterInflictConditionUnconscious   :: Bool

    , monsterMultiAttack                   :: Bool
    , monsterSpellcasting                  :: Bool

    , monsterDamageTags                    :: [String]
    , monsterSpellCastingTags              :: [String]
    , monsterTraitTags                     :: [String]

    , monsterEloRank                       :: Int

    }
    deriving stock    (Eq, Generic, Ord, Show)


instance DefaultOrdered StatBlock where

    headerOrder = const $ fromList
        [ "Name"
        , "Type"
        , "Size"
        , "Armor"
        , "Hit Points"
        , "Move Burrow"
        , "Move Climb"
        , "Move Fly"
        , "Move Swim"
        , "Move Walk"
        , "Stat Str"
        , "Stat Dex"
        , "Stat Con"
        , "Stat Int"
        , "Stat Wis"
        , "Stat Cha"
        , "Save Str"
        , "Save Dex"
        , "Save Con"
        , "Save Int"
        , "Save Wis"
        , "Save Cha"
        , "Blind Sight"
        , "Dark Vision"
        , "Tremorsense"
        , "True Sight"
        , "Immune Acid"
        , "Immune Bludgeoning"
        , "Immune Cold"
        , "Immune Fire"
        , "Immune Force"
        , "Immune Lightning"
        , "Immune Necrotic"
        , "Immune Piercing"
        , "Immune Poison"
        , "Immune Psychic"
        , "Immune Radiant"
        , "Immune Slashing"
        , "Immune Thunder"
        , "Resist Acid"
        , "Resist Bludgeoning"
        , "Resist Cold"
        , "Resist Fire"
        , "Resist Force"
        , "Resist Lightning"
        , "Resist Necrotic"
        , "Resist Piercing"
        , "Resist Poison"
        , "Resist Psychic"
        , "Resist Radiant"
        , "Resist Slashing"
        , "Resist Thunder"
        , "Cause Blinded"
        , "Cause Charmed"
        , "Cause Deafened"
        , "Cause Frightened"
        , "Cause Grappled"
        , "Cause Incapacitated"
        , "Cause Invisible"
        , "Cause Paralyzed"
        , "Cause Petrified"
        , "Cause Poisoned"
        , "Cause Prone"
        , "Cause Restrained"
        , "Cause Stunned"
        , "Cause Unconscious"
        , "Multiattack"
        , "Spellcasting"
        , "Damage Tags"
        , "Spellcasting Tags"
        , "Trait Tags"
        , "Elo Rank"
        ]


instance FromJSON Bonus where

    parseJSON = withText "Bonus" $ \txt ->
        let errMsg :: Parser a
            errMsg = fail $ "Could not parse Bonus: '" <> unpack txt <> "'"
        in  maybe errMsg pure $ uncons txt >>= parseBonus
      where
        parseBonus (sign, val) = do
            f <- case sign of
                  '+' -> Just id
                  '-' -> Just negate
                  _   -> Nothing
            v <- readMaybe . takeWhile isDigit $ unpack val :: Maybe Word
            pure . Bonus . f $ fromEnum v


instance FromJSON MonsterSize where

    parseJSON = withArray "size" $ (fmap f . g)
      where
        g :: Array -> Parser Text
        g = let grab :: Value -> Parser Text
                grab = \case
                    String txt -> pure txt
                    _ -> empty
            in  grab . V.head

        f :: Text -> MonsterSize
        f = let infer = \case
                    []    -> Medium
                    'G':_ -> Giant
                    'H':_ -> Huge
                    'L':_ -> Large
                    'M':_ -> Medium
                    'S':_ -> Small
                    'T':_ -> Tiny
                    _     -> Medium
            in  infer . unpack


data ModifierBlock = Junk


instance FromJSON ModifierBlock where

    parseJSON = withObject "Modifier Block" $ \obj -> do
        e <- obj .: "_copy" :: Parser Object
        pure $ e `seq` Junk


instance FromJSON StatBlock where

    parseJSON x = flip (withObject "Monster Stat-block") x $ \obj -> do
        monsterName                          <- obj .:? "name"   .!= "UNNAMED????"
        monsterSize                          <- obj .:? "size"   .!= Medium
        monsterType                          <- getType       obj
        monsterHitPoints                     <- getHitPoint   obj
        monsterArmor                         <- getArmorClass obj

        monsterSpeedBurrow                   <- obj `getSpeed` "burrow"
        monsterSpeedClimb                    <- obj `getSpeed` "climb"
        monsterSpeedFly                      <- obj `getSpeed` "fly"
        monsterSpeedSwim                     <- obj `getSpeed` "swim"
        monsterSpeedWalk                     <- obj `getSpeed` "walk"

        monsterStatStrength                  <- obj `getStat` "str"
        monsterStatDexterity                 <- obj `getStat` "dex"
        monsterStatConstitution              <- obj `getStat` "con"
        monsterStatIntelligence              <- obj `getStat` "int"
        monsterStatWisdom                    <- obj `getStat` "wis"
        monsterStatCharisma                  <- obj `getStat` "cha"

        monsterSaveStrength                  <- obj `getSave` "str"
        monsterSaveDexterity                 <- obj `getSave` "dex"
        monsterSaveConstitution              <- obj `getSave` "con"
        monsterSaveIntelligence              <- obj `getSave` "int"
        monsterSaveWisdom                    <- obj `getSave` "wis"
        monsterSaveCharisma                  <- obj `getSave` "cha"

        monsterBlindSight                    <- hasTagFrom obj "senseTags" ["B"]
        monsterDarkVision                    <- hasTagFrom obj "senseTags" ["D", "SD"]
        monsterTremorsense                   <- hasTagFrom obj "senseTags" ["T"]
        monsterTrueSight                     <- hasTagFrom obj "senseTags" ["U"]

        monsterDamageImmuneAcid              <- obj `hasImmunity` "acid"
        monsterDamageImmuneBludgeoning       <- obj `hasImmunity` "bludgeoning"
        monsterDamageImmuneCold              <- obj `hasImmunity` "cold"
        monsterDamageImmuneFire              <- obj `hasImmunity` "fire"
        monsterDamageImmuneForce             <- obj `hasImmunity` "force"
        monsterDamageImmuneLightning         <- obj `hasImmunity` "lightning"
        monsterDamageImmuneNecrotic          <- obj `hasImmunity` "necrotic"
        monsterDamageImmunePiercing          <- obj `hasImmunity` "piercing"
        monsterDamageImmunePoison            <- obj `hasImmunity` "poison"
        monsterDamageImmunePsychic           <- obj `hasImmunity` "psychic"
        monsterDamageImmuneRadiant           <- obj `hasImmunity` "radiant"
        monsterDamageImmuneSlashing          <- obj `hasImmunity` "slashing"
        monsterDamageImmuneThunder           <- obj `hasImmunity` "thunder"

        monsterDamageResistAcid              <- obj `hasResist` "acid"
        monsterDamageResistBludgeoning       <- obj `hasResist` "bludgeoning"
        monsterDamageResistCold              <- obj `hasResist` "cold"
        monsterDamageResistFire              <- obj `hasResist` "fire"
        monsterDamageResistForce             <- obj `hasResist` "force"
        monsterDamageResistLightning         <- obj `hasResist` "lightning"
        monsterDamageResistNecrotic          <- obj `hasResist` "necrotic"
        monsterDamageResistPiercing          <- obj `hasResist` "piercing"
        monsterDamageResistPoison            <- obj `hasResist` "poison"
        monsterDamageResistPsychic           <- obj `hasResist` "psychic"
        monsterDamageResistRadiant           <- obj `hasResist` "radiant"
        monsterDamageResistSlashing          <- obj `hasResist` "slashing"
        monsterDamageResistThunder           <- obj `hasResist` "thunder"

        monsterInflictConditionBlinded       <- obj `canInflictCondition` "blinded"
        monsterInflictConditionCharmed       <- obj `canInflictCondition` "charmed"
        monsterInflictConditionDeafened      <- obj `canInflictCondition` "deafened"
        monsterInflictConditionFrightened    <- obj `canInflictCondition` "frightened"
        monsterInflictConditionGrappled      <- obj `canInflictCondition` "grappled"
        monsterInflictConditionIncapacitated <- obj `canInflictCondition` "incapacitated"
        monsterInflictConditionInvisible     <- obj `canInflictCondition` "invisible"
        monsterInflictConditionParalyzed     <- obj `canInflictCondition` "paralyzed"
        monsterInflictConditionPetrified     <- obj `canInflictCondition` "petrified"
        monsterInflictConditionPoisoned      <- obj `canInflictCondition` "poisoned"
        monsterInflictConditionProne         <- obj `canInflictCondition` "prone"
        monsterInflictConditionRestrained    <- obj `canInflictCondition` "restrained"
        monsterInflictConditionStunned       <- obj `canInflictCondition` "stunned"
        monsterInflictConditionUnconscious   <- obj `canInflictCondition` "unconscious"

        monsterMultiAttack                   <- canMultiAttack obj
        monsterSpellcasting                  <- canCastSpells  obj

        monsterDamageTags                    <- getTagList obj "damageTags"
        monsterSpellCastingTags              <- getTagList obj "spellcastingTags"
        monsterTraitTags                     <- getTagList obj "traitTags"

        pure $ StatBlock
            { monsterName                          = monsterName
            , monsterType                          = monsterType
            , monsterSize                          = monsterSize
            , monsterArmor                         = monsterArmor
            , monsterHitPoints                     = monsterHitPoints
            , monsterSpeedBurrow                   = monsterSpeedBurrow
            , monsterSpeedClimb                    = monsterSpeedClimb
            , monsterSpeedFly                      = monsterSpeedFly
            , monsterSpeedSwim                     = monsterSpeedSwim
            , monsterSpeedWalk                     = monsterSpeedWalk
            , monsterStatStrength                  = monsterStatStrength
            , monsterStatDexterity                 = monsterStatDexterity
            , monsterStatConstitution              = monsterStatConstitution
            , monsterStatIntelligence              = monsterStatIntelligence
            , monsterStatWisdom                    = monsterStatWisdom
            , monsterStatCharisma                  = monsterStatCharisma
            , monsterSaveStrength                  = monsterSaveStrength
            , monsterSaveDexterity                 = monsterSaveDexterity
            , monsterSaveConstitution              = monsterSaveConstitution
            , monsterSaveIntelligence              = monsterSaveIntelligence
            , monsterSaveWisdom                    = monsterSaveWisdom
            , monsterSaveCharisma                  = monsterSaveCharisma
            , monsterBlindSight                    = monsterBlindSight
            , monsterDarkVision                    = monsterDarkVision
            , monsterTremorsense                   = monsterTremorsense
            , monsterTrueSight                     = monsterTrueSight
            , monsterDamageImmuneAcid              = monsterDamageImmuneAcid
            , monsterDamageImmuneBludgeoning       = monsterDamageImmuneBludgeoning
            , monsterDamageImmuneCold              = monsterDamageImmuneCold
            , monsterDamageImmuneFire              = monsterDamageImmuneFire
            , monsterDamageImmuneForce             = monsterDamageImmuneForce
            , monsterDamageImmuneLightning         = monsterDamageImmuneLightning
            , monsterDamageImmuneNecrotic          = monsterDamageImmuneNecrotic
            , monsterDamageImmunePiercing          = monsterDamageImmunePiercing
            , monsterDamageImmunePoison            = monsterDamageImmunePoison
            , monsterDamageImmunePsychic           = monsterDamageImmunePsychic
            , monsterDamageImmuneRadiant           = monsterDamageImmuneRadiant
            , monsterDamageImmuneSlashing          = monsterDamageImmuneSlashing
            , monsterDamageImmuneThunder           = monsterDamageImmuneThunder
            , monsterDamageResistAcid              = monsterDamageResistAcid
            , monsterDamageResistBludgeoning       = monsterDamageResistBludgeoning
            , monsterDamageResistCold              = monsterDamageResistCold
            , monsterDamageResistFire              = monsterDamageResistFire
            , monsterDamageResistForce             = monsterDamageResistForce
            , monsterDamageResistLightning         = monsterDamageResistLightning
            , monsterDamageResistNecrotic          = monsterDamageResistNecrotic
            , monsterDamageResistPiercing          = monsterDamageResistPiercing
            , monsterDamageResistPoison            = monsterDamageResistPoison
            , monsterDamageResistPsychic           = monsterDamageResistPsychic
            , monsterDamageResistRadiant           = monsterDamageResistRadiant
            , monsterDamageResistSlashing          = monsterDamageResistSlashing
            , monsterDamageResistThunder           = monsterDamageResistThunder
            , monsterInflictConditionBlinded       = monsterInflictConditionBlinded
            , monsterInflictConditionCharmed       = monsterInflictConditionCharmed
            , monsterInflictConditionDeafened      = monsterInflictConditionDeafened
            , monsterInflictConditionFrightened    = monsterInflictConditionFrightened
            , monsterInflictConditionGrappled      = monsterInflictConditionGrappled
            , monsterInflictConditionIncapacitated = monsterInflictConditionIncapacitated
            , monsterInflictConditionInvisible     = monsterInflictConditionInvisible
            , monsterInflictConditionParalyzed     = monsterInflictConditionParalyzed
            , monsterInflictConditionPetrified     = monsterInflictConditionPetrified
            , monsterInflictConditionPoisoned      = monsterInflictConditionPoisoned
            , monsterInflictConditionProne         = monsterInflictConditionProne
            , monsterInflictConditionRestrained    = monsterInflictConditionRestrained
            , monsterInflictConditionStunned       = monsterInflictConditionStunned
            , monsterInflictConditionUnconscious   = monsterInflictConditionUnconscious
            , monsterMultiAttack                   = monsterMultiAttack
            , monsterSpellcasting                  = monsterSpellcasting
            , monsterDamageTags                    = monsterDamageTags
            , monsterSpellCastingTags              = monsterSpellCastingTags
            , monsterTraitTags                     = monsterTraitTags
            , monsterEloRank                       = 0
            }


instance ToField Bonus where

    toField = fromString . show . fromEnum


instance ToField MonsterSize where

    toField = fromString . show . fromEnum


instance ToNamedRecord StatBlock where

    toNamedRecord StatBlock{..} = namedRecord
        [ "Name"                .= monsterName
        , "Type"                .= monsterType
        , "Size"                .= monsterSize
        , "Armor"               .= monsterArmor
        , "Hit Points"          .= monsterHitPoints
        , "Move Burrow"         .= monsterSpeedBurrow
        , "Move Climb"          .= monsterSpeedClimb
        , "Move Fly"            .= monsterSpeedFly
        , "Move Swim"           .= monsterSpeedSwim
        , "Move Walk"           .= monsterSpeedWalk
        , "Stat Str"            .= monsterStatStrength
        , "Stat Dex"            .= monsterStatDexterity
        , "Stat Con"            .= monsterStatConstitution
        , "Stat Int"            .= monsterStatIntelligence
        , "Stat Wis"            .= monsterStatWisdom
        , "Stat Cha"            .= monsterStatCharisma
        , "Save Str"            .= monsterSaveStrength
        , "Save Dex"            .= monsterSaveDexterity
        , "Save Con"            .= monsterSaveConstitution
        , "Save Int"            .= monsterSaveIntelligence
        , "Save Wis"            .= monsterSaveWisdom
        , "Save Cha"            .= monsterSaveCharisma
        , "Blind Sight"         .= boolToField monsterBlindSight
        , "Dark Vision"         .= boolToField monsterDarkVision
        , "Tremorsense"         .= boolToField monsterTremorsense
        , "True Sight"          .= boolToField monsterTrueSight
        , "Immune Acid"         .= boolToField monsterDamageImmuneAcid
        , "Immune Bludgeoning"  .= boolToField monsterDamageImmuneBludgeoning
        , "Immune Cold"         .= boolToField monsterDamageImmuneCold
        , "Immune Fire"         .= boolToField monsterDamageImmuneFire
        , "Immune Force"        .= boolToField monsterDamageImmuneForce
        , "Immune Lightning"    .= boolToField monsterDamageImmuneLightning
        , "Immune Necrotic"     .= boolToField monsterDamageImmuneNecrotic
        , "Immune Piercing"     .= boolToField monsterDamageImmunePiercing
        , "Immune Poison"       .= boolToField monsterDamageImmunePoison
        , "Immune Psychic"      .= boolToField monsterDamageImmunePsychic
        , "Immune Radiant"      .= boolToField monsterDamageImmuneRadiant
        , "Immune Slashing"     .= boolToField monsterDamageImmuneSlashing
        , "Immune Thunder"      .= boolToField monsterDamageImmuneThunder
        , "Resist Acid"         .= boolToField monsterDamageResistAcid
        , "Resist Bludgeoning"  .= boolToField monsterDamageResistBludgeoning
        , "Resist Cold"         .= boolToField monsterDamageResistCold
        , "Resist Fire"         .= boolToField monsterDamageResistFire
        , "Resist Force"        .= boolToField monsterDamageResistForce
        , "Resist Lightning"    .= boolToField monsterDamageResistLightning
        , "Resist Necrotic"     .= boolToField monsterDamageResistNecrotic
        , "Resist Piercing"     .= boolToField monsterDamageResistPiercing
        , "Resist Poison"       .= boolToField monsterDamageResistPoison
        , "Resist Psychic"      .= boolToField monsterDamageResistPsychic
        , "Resist Radiant"      .= boolToField monsterDamageResistRadiant
        , "Resist Slashing"     .= boolToField monsterDamageResistSlashing
        , "Resist Thunder"      .= boolToField monsterDamageResistThunder
        , "Cause Blinded"       .= boolToField monsterInflictConditionBlinded
        , "Cause Charmed"       .= boolToField monsterInflictConditionCharmed
        , "Cause Deafened"      .= boolToField monsterInflictConditionDeafened
        , "Cause Frightened"    .= boolToField monsterInflictConditionFrightened
        , "Cause Grappled"      .= boolToField monsterInflictConditionGrappled
        , "Cause Incapacitated" .= boolToField monsterInflictConditionIncapacitated
        , "Cause Invisible"     .= boolToField monsterInflictConditionInvisible
        , "Cause Paralyzed"     .= boolToField monsterInflictConditionParalyzed
        , "Cause Petrified"     .= boolToField monsterInflictConditionPetrified
        , "Cause Poisoned"      .= boolToField monsterInflictConditionPoisoned
        , "Cause Prone"         .= boolToField monsterInflictConditionProne
        , "Cause Restrained"    .= boolToField monsterInflictConditionRestrained
        , "Cause Stunned"       .= boolToField monsterInflictConditionStunned
        , "Cause Unconscious"   .= boolToField monsterInflictConditionUnconscious
        , "Multiattack"         .= boolToField monsterMultiAttack
        , "Spellcasting"        .= boolToField monsterSpellcasting
        , "Damage Tags"         .= collectionToField monsterDamageTags
        , "Spellcasting Tags"   .= collectionToField monsterSpellCastingTags
        , "Trait Tags"          .= collectionToField monsterTraitTags
        , "Elo Rank"            .= monsterEloRank
        ]


{-# SCC getType #-}
getType :: Object -> Parser String
getType obj = obj .:? key >>= maybe (pure def) (\v -> fromStr v <|> fromObj v)
  where
    key = "type"
    def = "UNTYPED"

    fromObj = withObject "Monster Type" $ \x -> x .:? key .!= def
    fromStr = withText   "Monster Type" $ pure . unpack


{-# SCC getHitPoint #-}
getHitPoint :: Object -> Parser Word
getHitPoint obj = obj .:? "hp" >>= maybe (pure 1) (withObject "Monster Type" $ \x -> x .:? "average" .!= 1)


{-# SCC getArmorClass #-}
getArmorClass :: Object -> Parser Word
getArmorClass obj = obj .:? "ac" .!= zed >>= fmap maximum . traverse extractFromEntry
  where
    extractFromEntry :: Value -> Parser Word
    extractFromEntry =
        \case
          Object v -> v .:? "ac"                .!= 0
          Number n -> pure (toBoundedInteger n) .!= 0
          _        -> pure 0


{-# SCC getSpeed #-}
getSpeed :: Object -> Key -> Parser Word
getSpeed obj k = obj .:? "speed" >>=
      maybe (pure 0) (withObject "SpeedList" $ \x -> x .:? k >>=
              maybe (pure 0) (\y -> fromObj y <|> fromNum y))
  where
    fromObj = withObject     "SpeedSubList" $ \z -> z .:? "number" .!= 0
    fromNum = withScientific "SpeedSubList" $ pure . sum . toBoundedInteger


{-# SCC getStat #-}
getStat :: Object -> Key -> Parser Word
getStat obj k = obj .:? k .!= 0


{-# SCC getSave #-}
getSave :: Object -> Key -> Parser Bonus
getSave obj k = obj .:? "save" >>= maybe (pure 0) (withObject "SaveList" $ \x -> x .:? k .!= 0)


{-# SCC hasImmunity #-}
hasImmunity :: Object -> String -> Parser Bool
hasImmunity obj k = obj .:? "immune" .!= zed >>= existence f
  where
    f x = textualEquality k x <|> withObject "immune" (`hasImmunity` k) x


{-# SCC hasResist #-}
hasResist :: Object -> String -> Parser Bool
hasResist obj k = (obj .:? "resist" .!= zed) >>= existence f
  where
    f x = textualEquality k x <|> withObject "resist" (`hasResist` k) x


{-# SCC hasTagFrom #-}
hasTagFrom :: Foldable f => Object -> Key -> f String -> Parser Bool
hasTagFrom obj label ks = obj .:? label .!= zed >>= existence (textualInclusion ks)


{-# SCC canInflictCondition #-}
canInflictCondition :: Object -> String -> Parser Bool
canInflictCondition obj k = (||) <$> f <*> g
  where
    f = obj .:? "conditionInflict"      .!= [] >>= existence (textualEquality k)
    g = obj .:? "conditionInflictSpell" .!= [] >>= existence (textualEquality k)


{-# SCC canMultiAttack #-}
canMultiAttack :: Object -> Parser Bool
canMultiAttack obj =
    obj .:? "actionTags" .!= zed >>= existence (textualEquality "Multiattack")


canCastSpells :: Object -> Parser Bool
canCastSpells obj = (obj .:? "spellcastingTags" .!= zed) <&> (not . null)


getTagList :: FromJSON a => Object -> Key -> Parser [a]
getTagList obj k = obj .:? k .!= []


{-# SCC textualEquality #-}
textualEquality :: String -> Value -> Parser Bool
textualEquality txt = withText txt $ pure . (== fromString txt)


textualInclusion :: Foldable f => f String -> Value -> Parser Bool
textualInclusion txts = withText (intercalate "," $ toList txts) $ pure . (`elem` txts) . unpack


{-# SCC existence #-}
existence :: Traversable t => (a -> Parser Bool) -> t a -> Parser Bool
existence f = fmap or . traverse f


zed :: Array
zed = mempty


collectionToField :: [String] -> Field
collectionToField = fromString . fold . intersperse ","


boolToField :: Bool -> Field
boolToField = fromString . show . fromEnum


setEloRank :: Int -> StatBlock -> StatBlock
setEloRank elo stat = stat { monsterEloRank = elo }


getMonsterName :: StatBlock -> Label
getMonsterName = Label . monsterName
