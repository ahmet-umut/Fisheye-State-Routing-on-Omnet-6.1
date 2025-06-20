// WirelessNode.cc - Enhanced Fish-eye State Routing implementation
// GPL v3 License - Advanced wireless simulation with realistic channel model

// Add this define at the top after the includes
#define VECTOR_COLLECTION 1  // Set to 1 for vector, 0 for scalar collection
// Add these includes at the top
#include <fstream>
#include <sstream>
#include <iomanip>

#include <omnetpp.h>
#include <map>
#include <vector>
#include <cmath>
#include <array>
#include <stack>
#include <queue>
#include <random>

using namespace omnetpp;
using namespace std;

// Maximum number of nodes - now configurable
#define NOC 66

//enum ConnState {UNKNOWN, CONNECTED, DISCONNECTED}; // Connection states
enum ConnState {DISCONNECTED, CONNECTED, UNKNOWN }; // Connection states
enum PacketType {PUBLISH, TOPOLOGY_UPDATE, DATA_FORWARD, HELLO, ACK, NACK};

struct ConnectionInfo {
    ConnState state = UNKNOWN;
    unsigned char age = 255;
    double linkQuality = 0.0;  // Link quality metric (0-1)
    simtime_t lastUpdate = 0;
};

struct ConnectionInfoMsg {
    unsigned char nodeId;
    std::array<ConnectionInfo, NOC> connections;

    ConnectionInfoMsg(unsigned char id, const std::array<ConnectionInfo, NOC>& conn)
        : nodeId(id), connections(conn) {}
};

struct Hop {
    int nextHop = -1;
    int distance = -1;
    bool reliable = false;
    double quality = 0.0;

    bool isValid() const { return nextHop != -1; }
    bool isUnreachable() const { return nextHop == -1; }
};

class FSRPacket : public cPacket {
public:
    PacketType packetType = PUBLISH;
    unsigned char sourceId, destId, nextHop;
    std::stack<ConnectionInfoMsg> connInfoStack;
    int sequenceNumber = 0;
    simtime_t timestamp;
    int hopCount = 0;
    double dataSize = 0;  // Data payload size in bytes

    FSRPacket(const char* name = nullptr) : cPacket(name) {
        sourceId = destId = nextHop = 0;
        timestamp = simTime();
    }
};

class WirelessNode : public cSimpleModule {
private:
    // Receiver state management
    bool receiverBusy = false;
    simtime_t receptionStartTime;
    simtime_t receptionEndTime;
    FSRPacket* currentlyReceivingPacket = nullptr;
    cMessage* receptionEndTimer = nullptr;

    // Queue for packets that arrive during reception
    std::queue<FSRPacket*> interferenceQueue;

    // Reception parameters
    double receptionThreshold = -85.0;  // dBm, minimum power to detect signal
    double captureThreshold = 3.0;      // dB, power difference needed for capture effect

    // Packet processing delays
    double packetProcessingDelay;  // Time to process a packet
    double transmissionDelay;      // Time to transmit based on packet size
    std::queue<std::pair<FSRPacket*, simtime_t>> packetQueue;  // Buffered packets
    cMessage *packetProcessingTimer;
    bool processingBusy = false;

    // Statistics file paths and scalar tracking
    std::string statsDir = "statistics/";
    std::string nodePrefix;///
#if VECTOR_COLLECTION
    void writeVectorStat(const std::string& statName, double value, bool ena=false);
#else
    // Scalar statistics tracking (only used if VECTOR_COLLECTION == 0)
    struct ScalarStat {
        double sum = 0.0;
        int count = 0;
        double mean = 0.0;

        void addValue(double value) {
            count++;
            sum += value;
            mean = sum / count;
        }
    };
    ScalarStat endToEndDelayStats;
    ScalarStat throughputStats;
    ScalarStat packetDeliveryRatioStats;
    ScalarStat controlOverheadStats;
    ScalarStat dataOverheadStats;
    ScalarStat energyConsumptionStats;
    void writeScalarStat(const std::string& statName, ScalarStat& stat, double value);
#endif
    void initializeStatFiles();

    // Node identity and position
    int myAddress;
    double myX, myY;
    double velocityX = 0, velocityY = 0;  // For mobility

    // Network parameters (now configurable)
    double transmissionRange;
    double transmissionPower;
    double dataRate;  // bits per second
    int maxNodes;

    // Wireless channel model parameters
    double pathLossExponent;
    double shadowingStdDev;
    double noiseFloor;
    double interferenceLevel;

    // FSR protocol parameters
    double fisheye_scope[3];  // Different scopes for fisheye
    std::array<bool, NOC> neighborUpToDate = {};
    std::array<bool, NOC> neighbors = {};
    std::array<std::array<ConnectionInfo, NOC>, NOC> topologyMatrix = {};
    std::array<Hop, NOC> nextHopTable;
    array<unsigned char, NOC> neista={};

    // Timers
    cMessage *helloTimer;
    cMessage *pubtim0,*pubtim1,*pubtim2,*pubtim3,*pubtimres;    double tis[9];
    cMessage *dataGenerationTimer;
    cMessage *mobilityTimer;
    cMessage *statisticsTimer;
    //cMessage *neitim;   double neitimout;
    cMessage *agetim;
    cMessage *conrectim;

    // Traffic generation
    double trafficGenerationRate;  // packets per second
    std::exponential_distribution<double> interArrivalTime;
    std::default_random_engine rng;
    int sequenceCounter = 0;

    // Performance tracking
    std::map<int, simtime_t> packetSentTimes;
    int totalPacketsSent = 0;
    int totalPacketsReceived = 0;
    int totalDataBitsTransmitted = 0;
    int totalControlBitsTransmitted = 0;
    int totalDataBitsDelivered = 0;
    double totalEnergyConsumed = 0.0;

    // Link quality and reliability
    std::map<int, double> linkReliability;
    double packetLossRate = 0.0;

protected:
    // New methods for receiver collision handling
    bool canReceivePacket(FSRPacket* packet, double receivedPower);
    void startReception(FSRPacket* packet, double receptionDuration);
    void endReception();
    void handleInterference(FSRPacket* packet);
    double calculateReceptionDuration(FSRPacket* packet);
    double calculateReceivedPowerFromPacket(FSRPacket* packet);

    virtual void initialize() override;
    virtual void handleMessage(cMessage *msg) override;
    virtual void finish() override;

    // Wireless communication
    double calculateDistance(double x1, double y1, double x2, double y2);
    double calculateReceivedPower(double txPower, double distance);
    double calculatePacketLossRate(double distance, double interference);
    bool transmitPacket(FSRPacket *packet, int destAddr = -1);
    void processReceivedPacket(FSRPacket *packet);

    // FSR Protocol
    void sendHello();
    void pub(short sev=0);
    void processHello(FSRPacket *packet);
    void processTopologyUpdate(FSRPacket *packet);
    void processDataPacket(FSRPacket *packet);

    // Routing calculations
    Hop calculateNextHop(int destination);
    void updateNextHopTable();
    void dijkstraWithUnknowns(int destination, bool useUnknowns,
                             std::array<int, NOC>& dist, std::array<int, NOC>& prev,
                             std::array<bool, NOC>& reliable);

    // Traffic generation and mobility
    void generateDataPacket();
    void updateMobility();

    // Statistics and monitoring
    void recordStatistics();
    void updateEnergyConsumption(double txPower, double duration);

    // Utility functions
    double getDistance(cModule* node);
    void broadcastPacket(FSRPacket* packet, double delay = 0, double duration = 0);
    bool isValidDestination(int dest);
};

Define_Module(WirelessNode);

void WirelessNode::initialize() {
    // Read processing parameters
    packetProcessingDelay = par("packetProcessingDelay");  // e.g., 0.001 seconds
    transmissionDelay = par("baseTransmissionDelay");      // e.g., 0.0001 seconds

    // Initialize processing timer
    packetProcessingTimer = new cMessage("packetProcessingTimer");

    // Read parameters
    myAddress = par("address");
    myX = par("x");
    myY = par("y");
    getDisplayString().setTagArg("p", 0, myX);
    getDisplayString().setTagArg("p", 1, myY);
    transmissionRange = par("transmissionRange");
    transmissionPower = par("transmissionPower");
    dataRate = par("dataRate");
    maxNodes = getParentModule()->par("numNodes");

    // Wireless channel parameters
    pathLossExponent = par("pathLossExponent");
    shadowingStdDev = par("shadowingStdDev");
    noiseFloor = par("noiseFloor");
    interferenceLevel = par("interferenceLevel");

    // FSR parameters
    fisheye_scope[0] = par("fisheyeScope1");
    fisheye_scope[1] = par("fisheyeScope2");
    fisheye_scope[2] = par("fisheyeScope3");

    // Traffic parameters
    trafficGenerationRate = par("trafficGenerationRate");
    packetLossRate = par("packetLossRate");

    // Initialize random number generator
    rng.seed(myAddress + intuniform(0, 1000000));
    interArrivalTime = std::exponential_distribution<double>(trafficGenerationRate);

    std::ostringstream oss;
    oss << "node_" << std::setfill('0') << std::setw(3) << myAddress << "_";
    nodePrefix = oss.str();
    // Initialize statistics files
    initializeStatFiles();

    // Initialize timers
    helloTimer = new cMessage("helloTimer",0);
    pubtim0 = new cMessage("pubtim",1);  agetim = new cMessage("agetim");
    pubtim1 = new cMessage("pubtim",2);
    pubtim2 = new cMessage("pubtim",3);
    pubtim3 = new cMessage("pubtim",4);
    pubtimres = new cMessage("pubtim",5);
    dataGenerationTimer = new cMessage("dataGenerationTimer");
    mobilityTimer = new cMessage("mobilityTimer");
    statisticsTimer = new cMessage("statisticsTimer");
    receptionEndTimer = new cMessage("receptionEndTimer");

    conrectim = new cMessage("conrectim");
    //neitim = new cMessage("neitim");

    // Schedule initial events with random delays to avoid synchronization
    double base = par("helloInterval").doubleValue() / log(par("maxSpeed").doubleValue() + 2);// / dataRate * (1<<14);// * NOC;
    tis[0]=base;
    scheduleAt(simTime() + base * uniform(0.9,1.1), helloTimer);
    scheduleAt(simTime() + (tis[1] = base = (base*2)) * uniform(0.9,1.1), pubtim0);
    scheduleAt(simTime() + (tis[2] = base = (base*2)) * uniform(0.9,1.1), pubtim1);
    scheduleAt(simTime() + (tis[3] = base = (base*2)) * uniform(0.9,1.1), pubtim2);
    scheduleAt(simTime() + (tis[4] = base = (base*2)) * uniform(0.9,1.1), pubtim3);
    scheduleAt(simTime() + (tis[5] = base = (base*2)) * uniform(0.9,1.1), pubtimres);
    //scheduleAt(simTime() + (neitimout) * uniform(0.9,1.1), neitim);
    scheduleAt(simTime() + (tis[0]/2) * uniform(0.9,1.1), agetim);

    scheduleAt(simTime() + (tis[0]) * uniform(0.9,1.1), conrectim);

    if (trafficGenerationRate > 0) {
        scheduleAt(simTime() + exponential(1.0/trafficGenerationRate), dataGenerationTimer);
    }

    if (par("mobilityEnabled").boolValue()) {
        scheduleAt(simTime() + par("mobilityUpdateInterval"), mobilityTimer);
    }

    scheduleAt(simTime() + par("statisticsInterval"), statisticsTimer);

    EV << "FSR Node " << myAddress << " initialized at (" << myX << "," << myY << ")" << endl;
}

void WirelessNode::handleMessage(cMessage *msg) {
    if (msg->isSelfMessage()) {
        if (msg == helloTimer) {
            sendHello();
            scheduleAt(simTime() + tis[0] * uniform(0.9, 1.1), helloTimer);
        }
        else if (msg == receptionEndTimer) {
            endReception();
            return;
        }
        else if (msg == pubtim0 || msg == pubtim1 || msg == pubtim2 || msg == pubtim3 || msg == pubtimres) {
            pub(msg != pubtimres ? msg->getKind()-1:-1);
            //double updateInterval = par("topologyUpdateInterval");
            scheduleAt(simTime() + tis[msg->getKind()] * uniform(0.9,1.1), msg);
        }
        else if (msg == agetim) {
            for(auto&row:topologyMatrix)for(auto&cell:row)if(cell.age!=255)    cell.age++;
        }
        else if (msg == conrectim) {    //record connectivity
            int con=0;  for (auto nei:neighbors)    con+=nei;
            writeVectorStat("connectivity", con);
            scheduleAfter(tis[0], msg);
        }
        else if (msg == dataGenerationTimer) {
            generateDataPacket();
            if (trafficGenerationRate > 0) {
                scheduleAt(simTime() + exponential(1.0/trafficGenerationRate), dataGenerationTimer);
            }
        }
        else if (msg == mobilityTimer) {
            updateMobility();
            scheduleAt(simTime() + par("mobilityUpdateInterval"), mobilityTimer);
        }
        else if (msg == statisticsTimer) {
            recordStatistics();
            scheduleAt(simTime() + par("statisticsInterval"), statisticsTimer);
        }
        else if (msg == packetProcessingTimer) {
            if (!packetQueue.empty()) {
                auto queuedPacket = packetQueue.front();
                packetQueue.pop();

                FSRPacket* packet = queuedPacket.first;

                // Calculate transmission delay based on packet size
                double txDelay = transmissionDelay + (packet->dataSize * 8) / dataRate;

                // Transmit with delay
                transmitPacket(packet, packet->nextHop);

                // Schedule next packet processing if queue not empty
                if (!packetQueue.empty()) {
                    scheduleAt(simTime() + packetProcessingDelay, packetProcessingTimer);
                } else {
                    processingBusy = false;
                }
            } else {
                processingBusy = false;
            }
        }

        return;
    }

    // Handle incoming packets with collision detection
    FSRPacket *packet = check_and_cast<FSRPacket*>(msg);

    // Calculate received power for this packet
    double receivedPower = calculateReceivedPowerFromPacket(packet);

    // Check if packet can be received
    if (canReceivePacket(packet, receivedPower)) {
        double receptionDuration = calculateReceptionDuration(packet);
        startReception(packet, receptionDuration);
    } else {
        // Packet is lost due to collision or insufficient power
        handleInterference(packet);
        delete packet;
    }
}
void WirelessNode::initializeStatFiles() {
    // Create statistics directory if it doesn't exist
    system(("mkdir " + statsDir + " 2>nul").c_str());


    // For vector collection, just create/clear files with headers
    std::vector<std::string> statNames = {
        "endToEndDelay", "throughput", "packetDeliveryRatio",
        "controlOverhead", "dataOverhead", "energyConsumption", "data_bits_transmitted", "data_bits_delivered", "control_bits_transmitted", "connectivity", "packet_sent", "packet_delivered"
    };

    for (const auto& statName : statNames) {
        std::ofstream file(statsDir + statName + ".txt");
        /*
        file << "# Vector Statistics for Node " << myAddress << std::endl;
        file << "# Format: timestamp value" << std::endl;
        */
        file.close();
    }
}
void WirelessNode::writeVectorStat(const std::string& statName, double value, bool ena) {
    std::string fna;
    if (ena)    fna=statName+".txt";
    else        fna=statsDir + statName + ".txt";
    std::ofstream file(fna, std::ios::app);
    file << std::fixed << std::setprecision(6) << simTime().dbl() << " " << value << std::endl;
    file.close();
}

double WirelessNode::calculateDistance(double x1, double y1, double x2, double y2) {
    return sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
}

double WirelessNode::calculateReceivedPower(double txPower, double distance) {
    if (distance < 1.0) distance = 1.0;  // Minimum distance

    // Path loss with shadowing
    double pathLoss = 40.0 + 10 * pathLossExponent * log10(distance);
    double shadowing = normal(0, shadowingStdDev);  // Log-normal shadowing
    double totalLoss = pathLoss + shadowing;

    // Convert to linear scale
    double receivedPower = txPower - totalLoss;
    return receivedPower;
}

double WirelessNode::calculatePacketLossRate(double distance, double interference) {
    double receivedPower = calculateReceivedPower(transmissionPower, distance);
    double snr = receivedPower - noiseFloor - interference;

    // Simple SNR-based packet loss model
    if (snr > 20) return 0.001;  // Very good
    if (snr > 15) return 0.01;   // Good
    if (snr > 10) return 0.05;   // Fair
    if (snr > 5) return 0.2;     // Poor
    return 0.8;  // Very poor
}

bool WirelessNode::transmitPacket(FSRPacket *packet, int destAddr) {
    packet->sourceId = myAddress;
    //packet->timestamp = simTime();

    // Calculate packet size for statistics
    double packetSize = 0;
    switch (packet->packetType) {
        case HELLO:
            packetSize = 64;  // bytes
            totalControlBitsTransmitted += packetSize * 8;
            break;
        case PUBLISH:
        case TOPOLOGY_UPDATE:
            packetSize = 128 + packet->connInfoStack.size() * (1 + maxNodes);
            totalControlBitsTransmitted += packetSize * 8;
            break;
        case DATA_FORWARD:
            packetSize = 64 + packet->dataSize;  // Header + payload
            totalDataBitsTransmitted += packetSize * 8;
            break;
    }

    // Energy consumption
    double transmissionTime = (packetSize * 8) / dataRate;
    updateEnergyConsumption(transmissionPower, transmissionTime);

    double txDelay = (packetSize * 8) / dataRate;

    if (destAddr == -1) {
        // Broadcast
        broadcastPacket(packet,txDelay);
        return true;
    } else {
        // Unicast - find the target node
        cModule *network = getParentModule();
        cModule *targetNode = network->getSubmodule("node", destAddr);
        if (targetNode) {
            double distance = getDistance(targetNode);
            double lossRate = calculatePacketLossRate(distance, interferenceLevel);

            if (distance <= transmissionRange && uniform(0, 1) > lossRate) {
                //FSRPacket *copy = new FSRPacket(*packet);
                //copy->timestamp = packet->timestamp;
                sendDirect(packet, txDelay, txDelay, targetNode, "radioIn$i");
                return true;
            }
        }
        return false;  // Transmission failed
    }
}

void WirelessNode::broadcastPacket(FSRPacket* packet, double delay, double duration) {
    cModule *network = getParentModule();
    int numNodes = network->par("numNodes");

    for (int i = 0; i < numNodes; i++) {
        if (i != myAddress) {
            cModule *node = network->getSubmodule("node", i);
            if (node) {
                double distance = getDistance(node);
                if (distance <= transmissionRange) {
                    double lossRate = calculatePacketLossRate(distance, interferenceLevel);
                    if (uniform(0, 1) > lossRate) {
                        FSRPacket *copy = new FSRPacket(*packet);
                        //copy->timestamp = packet->timestamp;
                        //cout << packet->timestamp << endl;
                        sendDirect(copy, delay, delay, node, "radioIn$i");
                    }
                }
            }
        }
    }
    delete packet;
}

void WirelessNode::processReceivedPacket(FSRPacket *packet) {
    switch (packet->packetType) {
        case HELLO:
            processHello(packet);
            break;
        case PUBLISH:
        case TOPOLOGY_UPDATE:
            processTopologyUpdate(packet);
            break;
        case DATA_FORWARD:
            processDataPacket(packet);
            break;
    }
    //delete packet;
}

void WirelessNode::sendHello() {
    FSRPacket *hello = new FSRPacket("HELLO");
    hello->packetType = HELLO;
    hello->sourceId = myAddress;

    EV << "Node " << myAddress << " sending HELLO" << endl;
    transmitPacket(hello);
}

void WirelessNode::processHello(FSRPacket *packet) {
    int senderId = packet->sourceId;
    bool cha=false;

    // Update neighbor information
    if (!neighbors[senderId]) {
        cha=true;
        neighbors[senderId] = true;
        //neighborUpToDate[senderId] = true;
        // Update topology matrix
        topologyMatrix[myAddress][senderId] = {CONNECTED, 0, 1.0, simTime()};
        topologyMatrix[senderId][myAddress] = {CONNECTED, 0, 1.0, simTime()};
        EV << "Node " << myAddress << " discovered neighbor " << senderId << endl;
    }

    //neighborUpToDate[senderId] = true;

    auto sta = ++neista[senderId];
    //unsigned char limsta = 255.0 / (par("maxSpeed")+1);
    if (sta==par("stalim").intValue()) for (int i=0; i<NOC; i++)  //between 16 and 32...
    {
        auto passta = neighbors[i];
        if (!neista[i])
        {
            neighbors[i]=0;
            topologyMatrix[myAddress][i] = {DISCONNECTED, 0, 1.0, simTime()};
            topologyMatrix[i][myAddress] = {DISCONNECTED, 0, 1.0, simTime()};
        }
        if (passta != neighbors[i])
        {
            cha=true;
        }
        neista[i]=0;
    }
    if (cha)
    {
        updateNextHopTable();
    }

    take(packet);
    delete(packet);
}

void WirelessNode::pub(short sev) {
    FSRPacket *update = new FSRPacket("TOPOLOGY_UPDATE");
    update->packetType = TOPOLOGY_UPDATE;
    update->sourceId = myAddress;

    array<unsigned char,NOC>ads={};
    ads[myAddress]=1;
    bool pick=false;
    if (sev==-1)
    {
        pick=true;
        sev = 3;
        while (sev--)
        {
            for (int i=0; i<NOC; i++)
            {
                if (ads[i]==1) for (int j=0; j<NOC; j++)
                {
                    auto c = topologyMatrix[i][j];
                    if (c.age!=255 && c.state)
                    {
                        ads[j]++;
                    }
                    ads[i]++;
                }
            }
        }
        for (auto&tra:ads)  tra = !tra;
    }
    else while (sev--)
    {
        for (int i=0; i<NOC; i++)
        {
            if (ads[i]==1) for (int j=0; j<NOC; j++)
            {
                auto c = topologyMatrix[i][j];
                if (c.age!=255 && c.state)
                {
                    ads[j]++;
                }
                ads[i]++;
            }
        }
    }
    update->connInfoStack = stack<ConnectionInfoMsg>();
    /*
    int count=0; for (auto tra:ads) if (tra==1) count++;
    if (count)    count = rand()%count;
    */
    int i=0;
    for (auto tra:ads)
    {
        if (tra==1) // && count==i)
        {
            update->connInfoStack.emplace(i,topologyMatrix[i]);
            //break;
        }
        i++;
    }
    if (update->connInfoStack.size())
        transmitPacket(update);

    //EV << "Node " << myAddress << " sending topology update with " << update->connInfoStack.size() << " entries" << endl;
}

void WirelessNode::processTopologyUpdate(FSRPacket *packet) {
    int senderId = packet->sourceId;
    bool topologyChanged = false;

    // Update neighbor status
    neighbors[senderId] = true;
    neighborUpToDate[senderId] = true;

    // Process topology information
    while (!packet->connInfoStack.empty()) {
        ConnectionInfoMsg connInfo = packet->connInfoStack.top();
        packet->connInfoStack.pop();

        int nodeId = connInfo.nodeId;
        if (nodeId != myAddress && nodeId < maxNodes) {
            for (int i = 0; i < maxNodes; i++) {
                if (connInfo.connections[i].age <= topologyMatrix[nodeId][i].age) {
                    topologyMatrix[nodeId][i] = connInfo.connections[i];
                    topologyMatrix[i][nodeId] = connInfo.connections[i];
                    topologyChanged = true;
                }
            }
        }
    }

    if (topologyChanged) {
        updateNextHopTable();
        EV << "Node " << myAddress << " updated topology from " << senderId << endl;
    }

    take(packet);
    delete packet;
}

void WirelessNode::generateDataPacket() {
    if (maxNodes <= 1) return;

    int destination;
    do {
        destination = intuniform(0, maxNodes - 1);
    } while (destination == myAddress);

    FSRPacket *dataPacket = new FSRPacket("DATA");
    dataPacket->packetType = DATA_FORWARD;
    dataPacket->sourceId = myAddress;
    dataPacket->destId = destination;
    dataPacket->sequenceNumber = ++sequenceCounter;
    dataPacket->dataSize = par("dataPacketSize").doubleValue();

    // Store for delay calculation
    int packetId = myAddress * 100000 + dataPacket->sequenceNumber;
    //packetSentTimes[packetId] = simTime();

    //cout << myAddress << " has generated a packet for " << destination << endl;

    Hop nextHop = nextHopTable[destination];
    if (nextHop.isValid()) {
        dataPacket->nextHop = nextHop.nextHop;
        EV << "Node " << myAddress << " generated data packet to " << destination
           << " via " << nextHop.nextHop << endl;
        transmitPacket(dataPacket, nextHop.nextHop);
        totalPacketsSent++;
    } else {
        EV << "Node " << myAddress << " cannot reach destination " << destination << endl;
        delete dataPacket;
    }
}

void WirelessNode::processDataPacket(FSRPacket *packet) {
    packet->hopCount++;

    if (packet->destId == myAddress) {
        // Packet reached destination
        cout << myAddress << " has received packet" << endl;
        totalPacketsReceived++;
        totalDataBitsDelivered += packet->dataSize * 8;

        // Calculate end-to-end delay
        int packetId = packet->sourceId * 100000 + packet->sequenceNumber;

        double delay = (simTime() - packet->timestamp).dbl();
        writeVectorStat("endToEndDelay", delay);
        writeVectorStat("throughput", packet->dataSize*8 / delay);
        take(packet);
        delete packet;
    } else {
        // Forward the packet
        Hop nextHop = nextHopTable[packet->destId];
        if (nextHop.isValid()) {
            packet->nextHop = nextHop.nextHop;
            // Add to processing queue instead of instant forward
            packetQueue.push({packet, simTime() + packetProcessingDelay});

            if (!processingBusy) {
                processingBusy = true;
                scheduleAt(simTime() + packetProcessingDelay, packetProcessingTimer);
            }
        } else {
            take(packet);
            delete packet;
        }
    }
}

Hop WirelessNode::calculateNextHop(int destination) {
    if (destination == myAddress || destination >= maxNodes || destination < 0) {
        return Hop();
    }

    std::array<int, NOC> dist;
    std::array<int, NOC> prev;
    std::array<bool, NOC> reliable;

    // Try reliable path first
    dijkstraWithUnknowns(destination, false, dist, prev, reliable);

    if (dist[destination] != INT_MAX) {
        int current = destination;
        while (prev[current] != myAddress && prev[current] != -1) {
            current = prev[current];
        }
        if (prev[current] == myAddress) {
            Hop result;
            result.nextHop = current;
            result.distance = dist[destination];
            result.reliable = true;
            result.quality = topologyMatrix[myAddress][current].linkQuality;
            return result;
        }
    }

    // Try with unknown connections
    dijkstraWithUnknowns(destination, true, dist, prev, reliable);

    if (dist[destination] != INT_MAX) {
        int current = destination;
        while (prev[current] != myAddress && prev[current] != -1) {
            current = prev[current];
        }
        if (prev[current] == myAddress) {
            Hop result;
            result.nextHop = current;
            result.distance = dist[destination];
            result.reliable = reliable[destination];
            result.quality = topologyMatrix[myAddress][current].linkQuality;
            return result;
        }
    }

    return Hop();
}
void WirelessNode::dijkstraWithUnknowns(int destination, bool useUnknowns,
                                        std::array<int, NOC>& dist, std::array<int, NOC>& prev,
                                        std::array<bool, NOC>& reliable) {
    // Initialize arrays
    for (int i = 0; i < maxNodes; i++) {
        dist[i] = INT_MAX;
        prev[i] = -1;
        reliable[i] = true;
    }
    dist[myAddress] = 0;

    // Priority queue for Dijkstra's algorithm
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>,
                       std::greater<std::pair<int, int>>> pq;
    pq.push({0, myAddress});

    std::array<bool, NOC> visited = {};

    // Store all possible predecessors for each node at the same distance
    std::array<std::vector<int>, NOC> allPredecessors;
    std::array<std::vector<bool>, NOC> allReliabilities;

    while (!pq.empty()) {
        int currentDist = pq.top().first;
        int currentNode = pq.top().second;
        pq.pop();

        if (visited[currentNode]) continue;
        visited[currentNode] = true;

        if (currentNode == destination) break;

        for (int neighbor = 0; neighbor < maxNodes; neighbor++) {
            if (neighbor == currentNode) continue;

            ConnState connection = topologyMatrix[currentNode][neighbor].state;
            bool canUse = false;
            bool isReliable = true;

            if (connection == CONNECTED) {
                canUse = true;
                isReliable = true;
            } else if (connection == UNKNOWN && useUnknowns) {
                canUse = true;
                isReliable = false;
            }

            if (canUse && !visited[neighbor]) {
                int newDist = currentDist + 1;
                bool newReliability = reliable[currentNode] && isReliable;

                if (newDist < dist[neighbor]) {
                    // Found a shorter path - clear previous predecessors and add this one
                    dist[neighbor] = newDist;
                    allPredecessors[neighbor].clear();
                    allReliabilities[neighbor].clear();
                    allPredecessors[neighbor].push_back(currentNode);
                    allReliabilities[neighbor].push_back(newReliability);
                    pq.push({newDist, neighbor});
                } else if (newDist == dist[neighbor]) {
                    // Found an equally good path - add to the list of predecessors
                    allPredecessors[neighbor].push_back(currentNode);
                    allReliabilities[neighbor].push_back(newReliability);
                }
            }
        }
    }

    // Now randomly select one predecessor for each node from all equally good options
    std::random_device rd;
    std::mt19937 gen(rd());

    for (int i = 0; i < maxNodes; i++) {
        if (!allPredecessors[i].empty()) {
            // Randomly select one of the equally good predecessors
            std::uniform_int_distribution<> dis(0, allPredecessors[i].size() - 1);
            int randomIndex = dis(gen);

            prev[i] = allPredecessors[i][randomIndex];
            reliable[i] = allReliabilities[i][randomIndex];
        }
    }
}

void WirelessNode::updateNextHopTable() {
    for (int dest = 0; dest < maxNodes; dest++) {
        if (dest != myAddress) {
            nextHopTable[dest] = calculateNextHop(dest);
        }
    }
}

void WirelessNode::updateMobility() {
    if (!par("mobilityEnabled").boolValue()) return;

    double maxSpeed = par("maxSpeed");
    double fieldX = getParentModule()->par("fieldX");
    double fieldY = getParentModule()->par("fieldY");

    // Random waypoint mobility model
    if (uniform(0, 1) < 0.1) {  // 10% chance to change direction
        double angle = uniform(0, 2 * M_PI);
        double speed = uniform(0, maxSpeed);
        velocityX = speed * cos(angle);
        velocityY = speed * sin(angle);
    }

    // Update position
    double newX = myX + velocityX * par("mobilityUpdateInterval").doubleValue();
    double newY = myY + velocityY * par("mobilityUpdateInterval").doubleValue();

    // Boundary reflection
    if (newX < 0 || newX > fieldX) velocityX = -velocityX;
    if (newY < 0 || newY > fieldY) velocityY = -velocityY;

    myX = std::max(0.0, std::min(fieldX, newX));
    myY = std::max(0.0, std::min(fieldY, newY));

    // Update display
    getDisplayString().setTagArg("p", 0, myX);
    getDisplayString().setTagArg("p", 1, myY);
}

// In recordStatistics(), replace all emit() calls:
void WirelessNode::recordStatistics() {
    // Calculate packet delivery ratio
    double pdr = totalPacketsSent > 0 ? (double)totalPacketsReceived / totalPacketsSent : 0;
    writeVectorStat("packetDeliveryRatio", pdr);

    // Calculate control overhead
    double controlOverhead = totalDataBitsDelivered > 0 ?
                            (double)totalControlBitsTransmitted / totalDataBitsDelivered : 0;
    writeVectorStat("controlOverhead", controlOverhead);

    // Calculate data overhead
    double dataOverhead = totalDataBitsDelivered > 0 ?
                         (double)totalDataBitsTransmitted / totalDataBitsDelivered : 0;
    writeVectorStat("dataOverhead", dataOverhead);
    // Record energy consumption
    writeVectorStat("energyConsumption", totalEnergyConsumed);
}

void WirelessNode::updateEnergyConsumption(double txPower, double duration) {
    totalEnergyConsumed += txPower * duration / 1000.0;  // Convert mW*s to J
}

double WirelessNode::getDistance(cModule* node) {
    WirelessNode* wirelessNode = check_and_cast<WirelessNode*>(node);
    return calculateDistance(myX, myY, wirelessNode->myX, wirelessNode->myY);
}

// Implementation of new methods:

bool WirelessNode::canReceivePacket(FSRPacket* packet, double receivedPower) {
    // Check if signal is strong enough to be detected
    if (receivedPower < receptionThreshold) {
        EV << "Node " << myAddress << " - Packet from " << (int)packet->sourceId
           << " too weak: " << receivedPower << " dBm" << endl;
        return false;
    }

    // If receiver is not busy, packet can be received
    if (!receiverBusy) {
        return true;
    }

    // If receiver is busy, check for capture effect
    // Calculate power of currently receiving packet
    double currentPower = calculateReceivedPowerFromPacket(currentlyReceivingPacket);

    // Capture effect: if new packet is significantly stronger, it can capture the receiver
    if (receivedPower > currentPower + captureThreshold) {
        EV << "Node " << myAddress << " - Capture effect: new packet (" << receivedPower
           << " dBm) captures receiver from current packet (" << currentPower << " dBm)" << endl;

        // Cancel current reception
        if (currentlyReceivingPacket) {
            delete currentlyReceivingPacket;
            currentlyReceivingPacket = nullptr;
        }
        cancelEvent(receptionEndTimer);

        return true;
    }

    // Otherwise, collision occurs
    EV << "Node " << myAddress << " - Collision: receiver busy, packet lost" << endl;
    return false;
}

void WirelessNode::startReception(FSRPacket* packet, double receptionDuration) {
    receiverBusy = true;
    receptionStartTime = simTime();
    receptionEndTime = simTime() + receptionDuration;
    currentlyReceivingPacket = packet;

    // Schedule end of reception
    scheduleAt(receptionEndTime, receptionEndTimer);
}

void WirelessNode::endReception() {
    if (!receiverBusy || !currentlyReceivingPacket) {
        return;
    }

    EV << "Node " << myAddress << " - Completed reception of packet from "
       << (int)currentlyReceivingPacket->sourceId << endl;

    // Process the successfully received packet
    processReceivedPacket(currentlyReceivingPacket);

    // Reset receiver state
    receiverBusy = false;
    currentlyReceivingPacket = nullptr;
    receptionStartTime = 0;
    receptionEndTime = 0;
}

void WirelessNode::handleInterference(FSRPacket* packet) {
    // Log interference/collision
    EV << "Node " << myAddress << " - Packet from " << (int)packet->sourceId
       << " lost due to interference/collision" << endl;

    // If there's a packet currently being received, it might also be corrupted
    // depending on the interference model you want to implement

    // For a simple model, we can corrupt the current packet if interference is strong enough
    if (receiverBusy && currentlyReceivingPacket) {
        double currentPower = calculateReceivedPowerFromPacket(currentlyReceivingPacket);
        double interferePower = calculateReceivedPowerFromPacket(packet);

        // If interference is strong relative to current signal, corrupt current packet
        if (interferePower > currentPower - 6.0) {  // 6 dB threshold
            EV << "Node " << myAddress << " - Current reception corrupted by interference" << endl;

            // Cancel current reception
            delete currentlyReceivingPacket;
            currentlyReceivingPacket = nullptr;
            cancelEvent(receptionEndTimer);
            receiverBusy = false;
        }
    }
}

double WirelessNode::calculateReceptionDuration(FSRPacket* packet) {
    // Calculate packet size
    double packetSize = 0;
    switch (packet->packetType) {
        case HELLO:
            packetSize = 64;  // bytes
            break;
        case PUBLISH:
        case TOPOLOGY_UPDATE:
            packetSize = 128 + packet->connInfoStack.size() * (1 + maxNodes);
            break;
        case DATA_FORWARD:
            packetSize = 64 + packet->dataSize;  // Header + payload
            break;
    }

    //cout << packetSize << " bytes -> " << (packetSize * 8) / dataRate << " seconds" << endl;
    // Calculate transmission time based on data rate
    return (packetSize * 8) / dataRate;  // bits / (bits per second) = seconds
}

double WirelessNode::calculateReceivedPowerFromPacket(FSRPacket* packet) {
    if (!packet) return -999.0;

    // Find the source node to calculate distance
    cModule *network = getParentModule();
    cModule *sourceNode = network->getSubmodule("node", packet->sourceId);

    if (!sourceNode) return -999.0;

    double distance = getDistance(sourceNode);
    double mul = 1 + (packet->packetType==DATA_FORWARD);
    return calculateReceivedPower(transmissionPower*mul, distance);
}

void WirelessNode::finish() {
    recordStatistics();
    // Clean up receiver state
    if (currentlyReceivingPacket)
        delete currentlyReceivingPacket;

    writeVectorStat("data_bits_transmitted", totalDataBitsTransmitted);
    writeVectorStat("data_bits_delivered", totalDataBitsDelivered);
    writeVectorStat("control_bits_transmitted", totalControlBitsTransmitted);
    int con=0;  for (auto nei:neighbors)    con+=nei;
    writeVectorStat("connectivity", con);
    writeVectorStat("packet_sent", totalPacketsSent);
    writeVectorStat("packet_delivered", totalPacketsReceived);

    cancelAndDelete(helloTimer);
    cancelAndDelete(pubtim0);
    cancelAndDelete(pubtim1);
    cancelAndDelete(pubtim2);
    cancelAndDelete(pubtim3);
    cancelAndDelete(pubtimres);
    cancelAndDelete(dataGenerationTimer);
    cancelAndDelete(mobilityTimer);
    cancelAndDelete(statisticsTimer);
    cancelAndDelete(packetProcessingTimer);
    cancelAndDelete(conrectim);
    cancelAndDelete(receptionEndTimer);
    //noc connectivity mobility load delay
}
