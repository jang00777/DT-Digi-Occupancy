import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

INPUT = []

process = cms.Process('RECO', eras.Run2_2016)
process = cms.Process('HARVESTING')

process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.StandardSequences.DQMSaverAtRunEnd_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(INPUT),
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step3 nevts:100'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.DQMoutput = cms.OutputModule("DQMRootOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('DQMIO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('file:Reco.root'),
    outputCommands = process.DQMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition

process.dqmInfoDT = cms.EDAnalyzer("DQMEventInfo", subSystemFolder = cms.untracked.string('DT'))

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '80X_dataRun2_Prompt_v16', '')

#here the customization for the DT DQM begins
process.load("DQM.DTMonitorModule.dt_dqm_sourceclient_common_cff")

process.dtDQM_sequence = cms.Sequence(process.dtScalerInfoMonitor + process.gtDigis + process.reco + process.dtDQMTask + process.dqmInfoDT)

from FWCore.ParameterSet.Config import EDProducer as DQMEDHarvester
process.dtSummaryClients = DQMEDHarvester("DTSummaryClients")
process.dtHarversting = cms.Path(process.dtDQMTest + process.dtSummaryClients + process.DQMSaver)
 
# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.unpackers)
process.dqmoffline_step = cms.Path(process.physicsEventsFilter *  process.dtDQM_sequence)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step, process.dqmoffline_step, process.DQMoutput_step, process.dtHarversting)
