// SimpleRoutingExample.ned - Pure OMNeT++ routing protocol
// No INET dependencies required

package tp;

simple SimpleNode
{
    parameters:
        int address;  // Node address (0, 1, 2, 3...)
        
    gates:
        inout port[];
}

channel SimpleChannel extends ned.DelayChannel
{
    delay = 0.1ms;
}

network SimpleNetwork
{
    submodules:
        node[4]: SimpleNode {
            address = index;
        }
        
    connections:
        // Linear topology: 0-1-2-3
        node[0].port++ <--> SimpleChannel <--> node[1].port++;
        node[1].port++ <--> SimpleChannel <--> node[2].port++;
        node[2].port++ <--> SimpleChannel <--> node[3].port++;
}