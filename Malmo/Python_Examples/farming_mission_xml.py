mission_xml = '''<?xml version="1.0" encoding="UTF-8" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

              <About>
                <Summary>Setup Farm</Summary>
              </About>

		   <ModSettings>
		      <MsPerTick>1</MsPerTick>
		   </ModSettings>

              <ServerSection>
                <ServerInitialConditions>
                  <Time>
                    <StartTime>0</StartTime>
                    <AllowPassageOfTime>false</AllowPassageOfTime>
                  </Time>
                  <Weather>clear</Weather>
                </ServerInitialConditions>
                <ServerHandlers>
                  <FlatWorldGenerator generatorString="2;10x0;1;"/>
                  <DrawingDecorator>
                  </DrawingDecorator>
                  <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
              </ServerSection>

              <AgentSection mode="Survival">
                <Name>FarmerBot</Name>
                <AgentStart>
                  <Placement yaw="-90"/>
                  <Inventory>
                    <InventoryObject slot="0" type="wheat_seeds" quantity="64"/>
                    <InventoryObject slot="1" type="carrot" quantity="64"/>
                    <InventoryObject slot="2" type="potato" quantity="64"/>
                  </Inventory>
                </AgentStart>
                <AgentHandlers>
                  <ObservationFromFullStats/>
                  <ContinuousMovementCommands/>
                  <AbsoluteMovementCommands/>
                </AgentHandlers>
              </AgentSection>
            </Mission>'''
