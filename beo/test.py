import plyvel, pickle
import networkx as nx
db = plyvel.DB('/lab/tmpig4b/u/manhattan/data' + str(30) + '/')

with open('test.gpickle','rb') as f:
    G = pickle.load(f)

def str2byte(s):
    s = s.encode()
    return s

err = ['-4.798844087844643,-22.892889590573645', '-4.798844087844643,-22.892889590573645','-8.114668947703635,-18.56759091357786', '-8.334495479912874,-18.63917879046049', '-8.334495479912874,-18.63917879046049', '-8.342702668675429,-18.574553934308724', '-8.342702668675429,-18.574553934308724', '-8.342702668675429,-18.574553934308724', '-8.342702668675429,-18.574553934308724', '-8.342702668675429,-18.574553934308724', '-8.342702668675429,-18.574553934308724', '-8.342702668675429,-18.574553934308724', '-8.342702668675429,-18.574553934308724', '-8.342702668675429,-18.574553934308724', '-8.342702668675429,-18.574553934308724', '-8.342702668675429,-18.574553934308724', '-8.342702668675429,-18.574553934308724', '-8.342702668675429,-18.574553934308724', '-8.342702668675429,-18.574553934308724', '-8.342702668675429,-18.574553934308724', '-8.342702668675429,-18.574553934308724', '-8.342702668675429,-18.574553934308724', '-8.342702668675429,-18.574553934308724', '-8.342702668675429,-18.574553934308724', '-8.342702668675429,-18.574553934308724', '-8.342702668675429,-18.574553934308724', '-8.342702668675429,-18.574553934308724', '-8.342702668675429,-18.574553934308724', '-8.342702668675429,-18.574553934308724', '-8.342702668675429,-18.574553934308724', '-8.342702668675429,-18.574553934308724', '-8.342702668675429,-18.574553934308724', '-8.342702668675429,-18.574553934308724', '-8.342702668675429,-18.574553934308724', '-8.342702668675429,-18.574553934308724', '-8.342702668675429,-18.574553934308724', '-8.342702668675429,-18.574553934308724', '-8.342702668675429,-18.574553934308724', '-8.44606352477318,-18.28591203593585', '-8.44606352477318,-18.28591203593585', '-8.44606352477318,-18.28591203593585', '-8.44606352477318,-18.28591203593585', '-8.44606352477318,-18.28591203593585', '-8.44606352477318,-18.28591203593585', '-8.44606352477318,-18.28591203593585', '-8.342702668675429,-18.574553934308724', '-8.334495479912874,-18.63917879046049', '-8.334495479912874,-18.63917879046049', '-8.334495479912874,-18.63917879046049', '-8.334495479912874,-18.63917879046049', '-8.334495479912874,-18.63917879046049']

count=1
while True:
    if count>2000:
        break
    count+=1
    temp = db.get(str2byte(err[0]))
for i in err:
    temp = db.get(str2byte(i))
    print(len(temp))

#for pos in G.nodes():
    #pos = str(pos[0]) + ',' + str(pos[1])
    #if pos in err:
        #print('we get error')
    #temp = db.get(str2byte(pos))
    
print('ok')
