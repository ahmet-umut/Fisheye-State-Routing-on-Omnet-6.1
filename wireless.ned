// wireless.ned - Enhanced FSR Network Definition
// GPL v3 License

package tp;

simple WirelessNode
{
    parameters:
        // Node identification
        int address;
        double x @unit(m) = default(uniform(0m, parent.fieldX));
        double y @unit(m) = default(uniform(0m, parent.fieldY));
        
        int stalim = default(8);
        
        // Physical layer parameters
        double transmissionRange @unit(m) = default(100m);
        double transmissionPower @unit(mW) = default(20mW);
        double dataRate @unit(bps) = default(1Mbps);
        
        // Wireless channel model
        double pathLossExponent = default(2.0);      // 2.0 for free space, 3-4 for urban
        double shadowingStdDev @unit(dB) = default(4dB);
        double noiseFloor @unit(dBm) = default(-100dBm);
        double interferenceLevel @unit(dBm) = default(-90dBm);
        double packetLossRate = default(0.01);       // Base packet loss rate
        
        // FSR Protocol parameters
        double fisheyeScope1 = default(1);           // Closest neighbors
        double fisheyeScope2 = default(2);           // Medium distance
        double fisheyeScope3 = default(3);           // Furthest scope
        double helloInterval @unit(s) = default(1s);
        double topologyUpdateInterval @unit(s) = default(2s);
        
        //delays
        double packetProcessingDelay @unit(s) = default(1ms);
		double baseTransmissionDelay @unit(s) = default(0.1ms);
        
        // Traffic generation
        double trafficGenerationRate @unit(pps) = default(0.1pps);  // packets per second
        double dataPacketSize @unit(B) = default(512B);             // Data payload size
        
        // Mobility parameters
        bool mobilityEnabled = default(false);
        double maxSpeed @unit(mps) = default(5mps);                 // Maximum speed in m/s
        double mobilityUpdateInterval @unit(s) = default(0.1s);     // Mobility update frequency
        
        // Statistics collection
        double statisticsInterval @unit(s) = default(1s);           // Statistics recording interval
        
        // SIGNAL DECLARATIONS - Add these!
        //@signal[endToEndDelay](type=simtime_t);
        //@signal[throughput](type=double);
        //@signal[packetDeliveryRatio](type=double);
        //@signal[controlOverhead](type=double);
        //@signal[dataOverhead](type=double);
        //@signal[energyConsumption](type=double);
        
        @display("i=device/pc;is=m"); // PC icon
        //@statistic[endToEndDelay](title="End-to-End Delay"; source=endToEndDelay; record=mean,max,min,vector; interpolationmode=none);
        //@statistic[throughput](title="Throughput"; source=throughput; record=mean,max,min,vector; interpolationmode=none);
        //@statistic[packetDeliveryRatio](title="Packet Delivery Ratio"; source=packetDeliveryRatio; record=mean,max,min,vector; interpolationmode=none);
        //@statistic[controlOverhead](title="Control Overhead"; source=controlOverhead; record=mean,max,min,vector; interpolationmode=none);
        //@statistic[dataOverhead](title="Data Overhead"; source=dataOverhead; record=mean,max,min,vector; interpolationmode=none);
        //@statistic[energyConsumption](title="Energy Consumption"; source=energyConsumption; record=mean,max,min,vector; interpolationmode=none);

    gates:
        inout radioIn @directIn; // Wireless interface - receives from all
}

// Wireless channel with path loss model
channel WirelessChannel extends ned.DatarateChannel
{
    parameters:
        datarate = 1Mbps; // Wireless data rate
        delay = 0ms; // Propagation delay handled by distance
        double pathLossExponent = default(2.0); // Free space = 2, urban = 3-4
        double referenceDistance @unit(m) = default(1m);
        double referenceLoss @unit(dB) = default(40dB); // Loss at reference distance
}

network WirelessNetwork
{
    parameters:
        int numNodes = default(10);
        double fieldX @unit(m) = default(300m); // Field width
        double fieldY @unit(m) = default(300m); // Field height
        
        // Network topology parameters
        double nodeSpacing @unit(m) = default(50m);        // For grid topology
        string topologyType = default("random");           // "random", "grid", "line"
        
        // Simulation parameters
        double simulationDuration @unit(s) = default(300s);
        
        @display("bgl=2;bgb=800,600");

    submodules:
        node[numNodes]: WirelessNode {
            address = index;
            
            // Enhanced position calculation with multiple topology types
            x = parent.topologyType == "grid" ? 
                (index % int(sqrt(parent.numNodes))) * parent.nodeSpacing + uniform(0m, 10m) :
                
                parent.topologyType == "line" ?
                index * parent.nodeSpacing + uniform(-20m, 20m) :
                
                parent.topologyType == "star" ?
                (index == 0 ? parent.fieldX/2 : 
                 parent.fieldX/2 + (parent.fieldX/3) * cos(2 * 3.14159 * (index-1) / (parent.numNodes-1))) :
                
                parent.topologyType == "mesh" ?
                (index % int(sqrt(parent.numNodes))) * parent.nodeSpacing + uniform(-5m, 5m) :
                
                parent.topologyType == "ring" ?
                parent.fieldX/2 + (parent.fieldX/2.5) * cos(2 * 3.14159 * index / parent.numNodes) :
                
                parent.topologyType == "tree" ?
                (index == 0 ? parent.fieldX/2 :
                 index <= 3 ? parent.fieldX/2 + (index-1) * parent.nodeSpacing - parent.nodeSpacing :
                 parent.fieldX/2 + ((index-4) % 4 - 1.5) * parent.nodeSpacing/2) :
                
                parent.topologyType == "cluster" ?
                (int(index / 5) * parent.nodeSpacing * 2 + (index % 5) * parent.nodeSpacing/3 + uniform(-15m, 15m)) :
                
                parent.topologyType == "hexagon" ?
                (index < 6 ? parent.fieldX/2 + (parent.fieldX/3) * cos(2 * 3.14159 * index / 6) :
                 parent.fieldX/2 + uniform(-parent.fieldX/4, parent.fieldX/4)) :
                
                parent.topologyType == "cross" ?
                (index < parent.numNodes/2 ? parent.fieldX/2 + (index - parent.numNodes/4) * parent.nodeSpacing/2 :
                 (index - parent.numNodes/2) * parent.fieldX / (parent.numNodes/2)) :
                
                parent.topologyType == "double_line" ?
                (index % 2 == 0 ? (index/2) * parent.nodeSpacing : (index/2) * parent.nodeSpacing) :
                
                parent.topologyType == "spiral" ?
                parent.fieldX/2 + (index * parent.nodeSpacing/4) * cos(index * 0.5) :
                
                parent.topologyType == "torus" ?
                (index % int(sqrt(parent.numNodes))) * parent.nodeSpacing + uniform(-8m, 8m) :
                
                parent.topologyType == "butterfly" ?
                (index < parent.numNodes/2 ? 
                 parent.fieldX/4 + (index * parent.nodeSpacing/2) : 
                 parent.fieldX*3/4 + ((index - parent.numNodes/2) * parent.nodeSpacing/2)) :
                
                parent.topologyType == "diamond" ?
                (index == 0 ? parent.fieldX/2 :
                 index == 1 ? parent.fieldX/4 :
                 index == 2 ? parent.fieldX*3/4 :
                 parent.fieldX/2 + (index-3) * parent.nodeSpacing/3 + uniform(-10m, 10m)) :
                
                parent.topologyType == "hourglass" ?
                (index < parent.numNodes/3 ? 
                 parent.fieldX/2 + ((index - parent.numNodes/6) * parent.nodeSpacing/2) :
                 index < parent.numNodes*2/3 ?
                 parent.fieldX/2 + uniform(-parent.nodeSpacing/4, parent.nodeSpacing/4) :
                 parent.fieldX/2 + ((index - parent.numNodes*5/6) * parent.nodeSpacing/2)) :
                
                parent.topologyType == "pyramid" ?
                (index < 1 ? parent.fieldX/2 :
                 index < 4 ? parent.fieldX/2 + ((index-1) - 1) * parent.nodeSpacing/2 :
                 index < 10 ? parent.fieldX/2 + ((index-4) - 2.5) * parent.nodeSpacing/3 :
                 parent.fieldX/2 + ((index-10) - (parent.numNodes-10)/2.0) * parent.nodeSpacing/4) :
                
                parent.topologyType == "manhattan" ?
                ((index % int(sqrt(parent.numNodes))) * parent.nodeSpacing) :
                
                parent.topologyType == "hybrid_star" ?
                (index < 3 ? parent.fieldX/2 + (parent.fieldX/4) * cos(2 * 3.14159 * index / 3) :
                 index < 6 ? parent.fieldX/4 + (parent.fieldX/6) * cos(2 * 3.14159 * (index-3) / 3) :
                 parent.fieldX*3/4 + (parent.fieldX/6) * cos(2 * 3.14159 * (index-6) / 3)) :
                
                parent.topologyType == "binary_tree" ?
                (index == 0 ? parent.fieldX/2 :
                 int(log2(index+1)) <= 1 ? parent.fieldX/2 + (index*2-3) * parent.nodeSpacing/2 :
                 int(log2(index+1)) <= 2 ? parent.fieldX/2 + (index*2-7) * parent.nodeSpacing/4 :
                 parent.fieldX/2 + (index*2-15) * parent.nodeSpacing/8) :
                
                parent.topologyType == "dumbbell" ?
                (index < parent.numNodes/3 ? 
                 parent.fieldX/4 + (parent.fieldX/6) * cos(2 * 3.14159 * index / (parent.numNodes/3)) :
                 index < parent.numNodes*2/3 ?
                 parent.fieldX/2 + (index - parent.numNodes/3) * parent.nodeSpacing/4 :
                 parent.fieldX*3/4 + (parent.fieldX/6) * cos(2 * 3.14159 * (index-parent.numNodes*2/3) / (parent.numNodes/3))) :
                
                parent.topologyType == "bipartite" ?
                (index < parent.numNodes/2 ? 
                 (index * parent.nodeSpacing/2) :
                 parent.fieldX*1/4 + ((index - parent.numNodes/2) * parent.nodeSpacing/2)) :
                
                parent.topologyType == "wheel" ?
                (index == 0 ? parent.fieldX/2 :
                 parent.fieldX/2 + (parent.fieldX/3) * cos(2 * 3.14159 * (index-1) / (parent.numNodes-1))) :
                
                parent.topologyType == "petersen" ?
                (index < 5 ? parent.fieldX/2 + (parent.fieldX/3) * cos(2 * 3.14159 * index / 5) :
                 parent.fieldX/2 + (parent.fieldX/6) * cos(2 * 3.14159 * (index-5) / 5 + 3.14159/5)) :
                
                parent.topologyType == "small_world" ?
                (index * parent.fieldX / parent.numNodes + uniform(-parent.nodeSpacing/2, parent.nodeSpacing/2)) :
                
                parent.topologyType == "fractal" ?
                (index % 3 == 0 ? parent.fieldX/4 :
                 index % 3 == 1 ? parent.fieldX/2 :
                 parent.fieldX*3/4) + uniform(-parent.nodeSpacing/3, parent.nodeSpacing/3) :
                
                parent.topologyType == "lattice" ?
                ((index % int(sqrt(parent.numNodes))) * parent.nodeSpacing + 
                 (int(index / int(sqrt(parent.numNodes))) % 2) * parent.nodeSpacing/2) :
                
                uniform(0m, parent.fieldX);  // random (default)
                
            y = parent.topologyType == "grid" ?
                int(index / int(sqrt(parent.numNodes))) * parent.nodeSpacing + uniform(0m, 10m) :
                
                parent.topologyType == "line" ?
                parent.fieldY/2 + uniform(-50m, 50m) :
                
                parent.topologyType == "star" ?
                (index == 0 ? parent.fieldY/2 : 
                 parent.fieldY/2 + (parent.fieldY/3) * sin(2 * 3.14159 * (index-1) / (parent.numNodes-1))) :
                
                parent.topologyType == "mesh" ?
                int(index / int(sqrt(parent.numNodes))) * parent.nodeSpacing + uniform(0m, 5m) :
                
                parent.topologyType == "ring" ?
                parent.fieldY/2 + (parent.fieldY/3) * sin(2 * 3.14159 * index / parent.numNodes) :
                
                parent.topologyType == "tree" ?
                (index == 0 ? parent.fieldY/4 :
                 index <= 3 ? parent.fieldY/2 :
                 parent.fieldY * 3/4) :
                
                parent.topologyType == "cluster" ?
                (int(index / 5) * parent.nodeSpacing/2 + uniform(-10m, 10m)) :
                
                parent.topologyType == "hexagon" ?
                (index < 6 ? parent.fieldY/2 + (parent.fieldY/3) * sin(2 * 3.14159 * index / 6) :
                 parent.fieldY/2 + uniform(-parent.fieldY/4, parent.fieldY/4)) :
                
                parent.topologyType == "cross" ?
                (index < parent.numNodes/2 ? parent.fieldY/2 :
                 parent.fieldY/2 + ((index - parent.numNodes/2) - (parent.numNodes/4)) * parent.nodeSpacing/2) :
                
                parent.topologyType == "double_line" ?
                (index % 2 == 0 ? parent.fieldY/3 : parent.fieldY * 2/3) :
                
                parent.topologyType == "spiral" ?
                parent.fieldY/2 + (index * parent.nodeSpacing/4) * sin(index * 0.5) :
                
                parent.topologyType == "torus" ?
                (int(index / int(sqrt(parent.numNodes))) * parent.nodeSpacing + uniform(-8m, 8m)) :
                
                parent.topologyType == "butterfly" ?
                (index < parent.numNodes/2 ? 
                 parent.fieldY/3 + (index % 2) * parent.nodeSpacing/2 : 
                 parent.fieldY*2/3 + (int(index - parent.numNodes/2) % 2) * parent.nodeSpacing/2) :
                
                parent.topologyType == "diamond" ?
                (index == 0 ? parent.fieldY/4 :
                 index == 1 ? parent.fieldY/2 :
                 index == 2 ? parent.fieldY/2 :
                 parent.fieldY*3/4 + uniform(-10m, 10m)) :
                
                parent.topologyType == "hourglass" ?
                (index < parent.numNodes/3 ? 
                 parent.fieldY/4 :
                 index < parent.numNodes*2/3 ?
                 parent.fieldY/2 :
                 parent.fieldY*3/4) :
                
                parent.topologyType == "pyramid" ?
                (index < 1 ? parent.fieldY/4 :
                 index < 4 ? parent.fieldY/2 :
                 index < 10 ? parent.fieldY*2/3 :
                 parent.fieldY*3/4) :
                
                parent.topologyType == "manhattan" ?
                (int(index / int(sqrt(parent.numNodes))) * parent.nodeSpacing) :
                
                parent.topologyType == "hybrid_star" ?
                (index < 3 ? parent.fieldY/2 + (parent.fieldY/4) * sin(2 * 3.14159 * index / 3) :
                 index < 6 ? parent.fieldY/4 + (parent.fieldY/6) * sin(2 * 3.14159 * (index-3) / 3) :
                 parent.fieldY*3/4 + (parent.fieldY/6) * sin(2 * 3.14159 * (index-6) / 3)) :
                
                parent.topologyType == "binary_tree" ?
                (index == 0 ? parent.fieldY/4 :
                 int(log2(index+1)) <= 1 ? parent.fieldY/2 :
                 int(log2(index+1)) <= 2 ? parent.fieldY*2/3 :
                 parent.fieldY*3/4) :
                
                parent.topologyType == "dumbbell" ?
                (index < parent.numNodes/3 ? 
                 parent.fieldY/2 + (parent.fieldY/6) * sin(2 * 3.14159 * index / (parent.numNodes/3)) :
                 index < parent.numNodes*2/3 ?
                 parent.fieldY/2 :
                 parent.fieldY/2 + (parent.fieldY/6) * sin(2 * 3.14159 * (index-parent.numNodes*2/3) / (parent.numNodes/3))) :
                
                parent.topologyType == "bipartite" ?
                (index < parent.numNodes/2 ? 
                 parent.fieldY/3 + (index % 3) * parent.nodeSpacing/3 :
                 parent.fieldY*2/3 + (int(index - parent.numNodes/2) % 3) * parent.nodeSpacing/3) :
                
                parent.topologyType == "wheel" ?
                (index == 0 ? parent.fieldY/2 :
                 parent.fieldY/2 + (parent.fieldY/3) * sin(2 * 3.14159 * (index-1) / (parent.numNodes-1))) :
                
                parent.topologyType == "petersen" ?
                (index < 5 ? parent.fieldY/2 + (parent.fieldY/3) * sin(2 * 3.14159 * index / 5) :
                 parent.fieldY/2 + (parent.fieldY/6) * sin(2 * 3.14159 * (index-5) / 5 + 3.14159/5)) :
                
                parent.topologyType == "small_world" ?
                parent.fieldY/2 + (parent.fieldY/4) * sin(2 * 3.14159 * index / parent.numNodes) + uniform(-parent.nodeSpacing/3, parent.nodeSpacing/3) :
                
                parent.topologyType == "fractal" ?
                (int(index / 3) * parent.nodeSpacing/2 + uniform(-parent.nodeSpacing/4, parent.nodeSpacing/4)) :
                
                parent.topologyType == "lattice" ?
                (int(index / int(sqrt(parent.numNodes))) * parent.nodeSpacing) :
                
                uniform(0m, parent.fieldY);  // random (default)
        }

    connections allowunconnected:
        // Wireless connections are established dynamically based on range
        // No static connections needed - handled in C++ code
}