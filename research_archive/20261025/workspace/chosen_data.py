"""我手工生成的 chosen 结论（基于逐条读证据）。
每条: question, answer, supported_facts=[{fact, evidence_refs:[{evidence_id, quote}]}]
quote 必须严格复制自该条证据文本（程序会校验）。"""
CHOSEN = [
    # [0] 古米训练问题
    {
        "question": "古米在下午训练时表现出的问题具体是什么？坚雷教官如何看待这个问题？",
        "answer": "古米在战斗（追击赏金猎人）中最后一击时犹豫了，手顿住没有继续追击，导致自己本可避免地受了伤；这个关键时刻犹豫、不敢下重手的问题是她的习惯问题。坚雷教官已经注意到并向真理提起过古米的这个习惯问题，认为这在实战中很危险。",
        "supported_facts": [
            {"fact": "古米在战斗中最后一击时犹豫、手顿住没有继续追击，因此受了本可避免的伤。",
             "evidence_refs": [
                 {"evidence_id": "obt/memory/story_sunbr_2_1#chunk-0003",
                  "quote": "其实你是可以不受伤的吧？为什么没有继续追击？最后一击的时候，你的手顿住了，你在犹豫"}]},
            {"fact": "坚雷教官已经注意到古米的这个习惯问题，并向真理提起过。",
             "evidence_refs": [
                 {"evidence_id": "obt/memory/story_sunbr_2_1#chunk-0003",
                  "quote": "其实坚雷教官也和我提过你的习惯问题"}]},
        ],
    },
    # [1] 真理本名/学校关系 —— 证据池只含学校关系，本名未在证据中，故只确认可证部分
    {
        "question": "真理的本名是什么？她和古米、凛冬分别是什么学校的关系？",
        "answer": "根据现有证据可确认学校关系：古米（本名拉达）和凛冬（索尼娅）都是彼得海姆中学的学生，而真理与早露（娜塔莉娅）原属切尔诺伯格第四中学。证据未直接给出真理的本名。",
        "supported_facts": [
            {"fact": "古米本名拉达，是彼得海姆中学学生。",
             "evidence_refs": [
                 {"evidence_id": "moegirl/乌萨斯学生自治团-3bace29fcf11#chunk-0004",
                  "quote": "古米 本名拉达，彼得海姆中学学生"}]},
            {"fact": "凛冬原属彼得海姆中学，真理原属切尔诺伯格第四中学。",
             "evidence_refs": [
                 {"evidence_id": "moegirl/乌萨斯学生自治团-3bace29fcf11#chunk-0004",
                  "quote": "凛冬、烈夏等原属彼得海姆中学，早露、真理原属切尔诺伯格第四中学"}]},
        ],
    },
    # [2] 薇卡对真理的意义
    {
        "question": "薇卡对真理来说是什么人？为什么真理至今还保留着她的东西？",
        "answer": "薇卡是真理（安娜）在加入学生自治团之前最好的朋友。薇卡已经去世（或离去），但真理一直没有忘记她，把她当作至今仍陪在身边的存在，所以保留着她留下的东西（如那台录像器械所承载的回忆），靠它来自我倾诉、面对过去。",
        "supported_facts": [
            {"fact": "薇卡是真理在加入学生自治团之前最好的朋友。",
             "evidence_refs": [
                 {"evidence_id": "activities/act10d5/level_act10d5_st02#chunk-0001",
                  "quote": "薇卡她，是我在加入学生自治团前，最好的朋友"}]},
            {"fact": "真理至今没有忘记薇卡，感觉她仍陪在自己身边。",
             "evidence_refs": [
                 {"evidence_id": "activities/act10d5/level_act10d5_st02#chunk-0001",
                  "quote": "我从来没有忘记过，有关于你的事情"}]},
        ],
    },
    # [3] 学生如何被聚集到学校 —— 证据只确认"被聚集后差点内斗、娜塔莉娅组织"，过程未直接给出
    {
        "question": "整合运动占领切尔诺伯格时，学生们是如何被聚集到凛冬学校的？",
        "answer": "根据现有证据可确认：学生们被聚集到了同一所学校里，被聚集的第一天贵族学生之间就差点爆发流血事件，娜塔莉娅作为第四中学学生会长站出来，用她的威信把这些贵族学生组织了起来，最终将三十多人聚集在一起。证据未直接说明整合运动把学生集中到一起的具体执行过程。",
        "supported_facts": [
            {"fact": "学生被聚集到学校的第一天，贵族学生之间差点爆发流血事件。",
             "evidence_refs": [
                 {"evidence_id": "activities/act10d5/level_act10d5_st04#chunk-0015",
                  "quote": "在被聚集到学校里的第一天，贵族学生们之间就差点爆发流血事件"}]},
            {"fact": "娜塔莉娅作为学生会长站出来把贵族学生组织起来，最终聚集了三十多人。",
             "evidence_refs": [
                 {"evidence_id": "activities/act10d5/level_act10d5_st04#chunk-0015",
                  "quote": "我不得不作为第四中学的学生会长站出来，用我的威信将他们组织起来"}]},
        ],
    },
    # [4] 赫默传讯娜塔莉娅 —— 证据明确：是关于"测试"，需去医疗部
    {
        "question": "赫默医生给娜塔莉娅发送传讯的目的是什么？这与之前的什么测试有关？",
        "answer": "赫默医生给娜塔莉娅发传讯是让她去一趟医疗部，内容与娜塔莉娅之前提到的测试有关。证据中娜塔莉娅看到纸条后表示那是她之前提到的、关于测试的事情，并随即动身前往医疗部；但证据未进一步说明该测试的具体内容。",
        "supported_facts": [
            {"fact": "赫默医生给娜塔莉娅发传讯，让她去一趟医疗部。",
             "evidence_refs": [
                 {"evidence_id": "activities/act10d5/level_act10d5_st02#chunk-0002",
                  "quote": "这个纸条，上面写的是赫默小姐的传讯"}]},
            {"fact": "传讯与娜塔莉娅之前提到的测试有关。",
             "evidence_refs": [
                 {"evidence_id": "activities/act10d5/level_act10d5_st02#chunk-0002",
                  "quote": "是我之前提到的，关于测试的事情"}]},
        ],
    },
    # [5] 罗莎琳找索尼娅 —— 证据较弱，只能确认"她想加入索尼娅、并提到缺粮缺人"
    {
        "question": "罗莎琳为什么决定去找索尼娅？",
        "answer": "根据现有证据可确认：在被关押、粮食和人力都紧缺的情况下，罗莎琳主动去找索尼娅，表示想加入她，并提到两个食堂都没了、吃的根本不够，需要想个办法找出路。证据未明确给出她做出这一决定的完整动机。",
        "supported_facts": [
            {"fact": "罗莎琳主动去找索尼娅，表示想加入她。",
             "evidence_refs": [
                 {"evidence_id": "activities/act10d5/level_act10d5_st05#chunk-0013",
                  "quote": "我没有敌意，我只是想来，这个，呃，加入你的"}]},
            {"fact": "当时粮食紧缺，两个食堂都没了，罗莎琳认为需要想办法找出路。",
             "evidence_refs": [
                 {"evidence_id": "activities/act10d5/level_act10d5_st05#chunk-0016",
                  "quote": "但现在两个食堂都没了，吃的根本不够"}]},
        ],
    },
    # [6] 阿米娅与调岗申请 —— 证据只说阿米娅"不太想通过"，未说"同意"，如实只确认可证部分
    {
        "question": "阿米娅为什么改变主意同意娜塔莉娅的调岗申请？",
        "answer": "根据现有证据，娜塔莉娅申请从后勤部门调往前线，她是唯一在后勤部门工作的自治团成员、也是营救出的难民中唯一的贵族；阿米娅起初并不太想通过这份调岗申请，杜宾也理解阿米娅不想让这样的年轻人上战场的心情。但证据未直接说明阿米娅后来是否以及为何改变主意同意。",
        "supported_facts": [
            {"fact": "娜塔莉娅申请从后勤部门调往前线，阿米娅起初不太想通过。",
             "evidence_refs": [
                 {"evidence_id": "activities/act10d5/level_act10d5_st06#chunk-0001",
                  "quote": "意向是从后勤部门调往前线"}]},
            {"fact": "娜塔莉娅是营救出的难民中唯一的贵族，阿米娅不太想通过她的调岗申请。",
             "evidence_refs": [
                 {"evidence_id": "activities/act10d5/level_act10d5_st06#chunk-0001",
                  "quote": "这一次，我也不是很想通过呢"}]},
        ],
    },
    # [7] 烈夏醉蜜后 —— 证据：阿克托斯与烈夏相认、坦白身世
    {
        "question": "烈夏醉蜜后发生了什么？",
        "answer": "烈夏醉蜜后，在银心湖列车事件后续中与生父阿克托斯相认。阿克托斯向她坦白了身世：他娶了烈夏的母亲，而在烈夏出生后不久希瓦艾什夫妇就遭遇了并非单纯意外的变故。烈夏一时难以接受，拒绝原谅这个十几年未曾出现的父亲。",
        "supported_facts": [
            {"fact": "烈夏在雪地中与阿克托斯相认。",
             "evidence_refs": [
                 {"evidence_id": "activities/act30side/level_act30side_06_end#chunk-0013",
                  "quote": "雪地中，阿克托斯与烈夏相认"}]},
            {"fact": "阿克托斯告知烈夏，她出生后不久希瓦艾什夫妇就出了意外。",
             "evidence_refs": [
                 {"evidence_id": "activities/act30side/level_act30side_06_end#chunk-0014",
                  "quote": "在你出生后没多久，希瓦艾什夫妇就出了意外"}]},
        ],
    },
    # [8] 罗莎琳等人的困境 —— 证据：粮食点被烧毁、学生挨饿、逃出校园即送死
    {
        "question": "整合运动占领学校期间，罗莎琳等人面临哪些困境？",
        "answer": "罗莎琳等人面临的困境包括：粮食严重短缺，第二场大火烧毁了学校里最后一个粮食点，学生接连挨饿、为抢食物互相争斗；同时学校被整合运动控制，一旦私自逃出受控制的校园就等于送死，学生们被困在其中。",
        "supported_facts": [
            {"fact": "大火烧毁了学校最后一个粮食点，学生挨饿、为抢食物争斗。",
             "evidence_refs": [
                 {"evidence_id": "activities/act10d5/level_act10d5_st03#chunk-0021",
                  "quote": "第二场大火将贵族杂碎们的据点粉碎，但也让学校里的最后一个粮食点消失"}]},
            {"fact": "学生一旦逃出受控制的校园就等于送死。",
             "evidence_refs": [
                 {"evidence_id": "activities/act10d5/level_act10d5_st07#chunk-0013",
                  "quote": "出去，可就是自己送死了"}]},
        ],
    },
    # [12] 整合运动对梅菲斯特的态度 —— 证据充分
    {
        "question": "整合运动内部对梅菲斯特的态度如何，为什么部分成员对他不满？",
        "answer": "整合运动内部不少成员对梅菲斯特极为不满，认为他是疯子、怪物，甚至有人想杀了他（只是被大尉禁止）。不满的原因是他纵容手下乱来、使用恶毒的源石技艺，做出闯进别人地盘杀人放火的暴徒行径，与整合运动的初衷相违背。",
        "supported_facts": [
            {"fact": "有成员认为梅菲斯特是疯子、怪物，想杀了他（但被大尉禁止）。",
             "evidence_refs": [
                 {"evidence_id": "activities/act10d5/level_act10d5_st07#chunk-0003",
                  "quote": "有时候我真想杀了梅菲斯特，如果不是大尉不准，我保准现在就去杀了他"}]},
            {"fact": "成员不满他纵容手下乱来、做杀人放火的暴徒行径。",
             "evidence_refs": [
                 {"evidence_id": "activities/act10d5/level_act10d5_st07#chunk-0003",
                  "quote": "就这么纵容他手下那些人乱来，我们是这种闯进别人家杀人放火的暴徒吗"}]},
        ],
    },
    # [13] 感染者社区为何频发暴力 —— 证据：媒体污名化、栽赃煽动
    {
        "question": "为什么感染者社区会频繁遭受针对感染者的暴力事件？",
        "answer": "卡拉顿的感染者社区频遭暴力，一方面是因为《卡拉顿日报》等媒体长期用耸人听闻的标题污名化感染者、煽动对立；另一方面有人刻意伪装成受害者栽赃陷害（如同一人先后伪装成工厂倒闭的工人、被偷的商贩混入示威队伍），煽动市民围攻、驱逐感染者。",
        "supported_facts": [
            {"fact": "媒体用耸人听闻的标题污名化感染者、煽动对立。",
             "evidence_refs": [
                 {"evidence_id": "activities/act15side/level_act15side_st01#chunk-0003",
                  "quote": "先整个耸人听闻的标题，再把“感染者”三个字塞进去"}]},
            {"fact": "有人刻意伪装成受害者混入示威队伍，栽赃煽动市民驱逐感染者。",
             "evidence_refs": [
                 {"evidence_id": "activities/act15side/level_act15side_st01#chunk-0001",
                  "quote": "他穿着连体工服，混在示威者的队伍里，说自己的工厂因为感染者倒闭了"}]},
        ],
    },
    # [14] 店铺归属 + 罗德岛角色 —— 证据明确
    {
        "question": "店铺实际上是归属于谁的？罗德岛在感染者社区扮演什么角色？",
        "answer": "这家店实际上是罗德岛的资产。罗德岛在卡拉顿的感染者社区扮演着医疗援助者的角色：它设立医疗站，为感染者提供矿石病诊断、急性发作救治乃至普通工伤处理，用实际行动逐渐赢得了社区里一部分感染者的信任。",
        "supported_facts": [
            {"fact": "这家店其实是罗德岛的资产。",
             "evidence_refs": [
                 {"evidence_id": "activities/act15side/level_act15side_st01#chunk-0004",
                  "quote": "这家店其实是罗德岛的资产"}]},
            {"fact": "罗德岛在感染者社区提供矿石病诊断、救治等医疗帮助，赢得部分感染者信任。",
             "evidence_refs": [
                 {"evidence_id": "activities/act15side/level_act15side_st01#chunk-0001",
                  "quote": "矿石病病情的诊断，矿石病急性发作的救治，甚至是普通工伤的处理，只要去了罗德岛医疗站，都能得到帮助"}]},
        ],
    },
    # [15] 苏茜为何在废弃城区遇夜烟 —— 证据：夜烟误入旧仓库探查，苏茜去废弃城区找人/查线索
    {
        "question": "苏茜为什么会在废弃城区遇到夜烟？",
        "answer": "根据现有证据可确认：夜烟是因为靠近废弃城区的一栋旧仓库无人居住、想进去看看有没有能用的东西，才来到废弃城区；两人是在夜烟误入那栋旧仓库、撞破绑架者密谋时相遇的。证据未直接说明苏茜当时前往废弃城区的具体原因。",
        "supported_facts": [
            {"fact": "夜烟因旧仓库无人居住、想进去找能用的东西而来到废弃城区。",
             "evidence_refs": [
                 {"evidence_id": "activities/act15side/level_act15side_st04#chunk-0005",
                  "quote": "这边的建筑物，一直没有人住啊......是因为靠近废弃城区吗"}]},
            {"fact": "夜烟在旧仓库撞见了绑架者密谋。",
             "evidence_refs": [
                 {"evidence_id": "activities/act15side/level_act15side_st04#chunk-0005",
                  "quote": "不就是绑架几个感染者和一个老头子吗，至于这么兴师动众的"}]},
        ],
    },
    # [16] 天火为何解救被绑架感染者 —— 证据：她在现场、阻止事态恶化
    {
        "question": "天火为什么要出手解救被绑架的感染者？",
        "answer": "根据现有证据可确认：有人在感染者社区绑架感染者（还涉及炸工厂、绑架城镇议员），天火当时正好在现场，出手制止才没让事态恶化。证据未明确说明她出手的全部动机，但表明她是恰好在场并阻止了绑架。",
        "supported_facts": [
            {"fact": "有人在感染者社区尝试绑架（议员/感染者），天火在现场制止，才没让事态恶化。",
             "evidence_refs": [
                 {"evidence_id": "activities/act15side/level_act15side_st02#chunk-0002",
                  "quote": "有人尝试绑架城镇议员"}]},
            {"fact": "天火在绑架现场出手阻止。",
             "evidence_refs": [
                 {"evidence_id": "activities/act15side/level_act15side_st02#chunk-0002",
                  "quote": "不过好在天火在现场，才没有让事态恶化"}]},
        ],
    },
    # [17] 夜烟如何逃脱火 —— 证据：用源石技艺造黑雾迷惑感知、带伤逃走
    {
        "question": "夜烟是如何逃脱黑衣人在废弃建筑放的火的？",
        "answer": "夜烟靠她的源石技艺制造黑雾迷惑黑衣人的感知，让对方难以锁定和击中她，从而带着箭伤逃出。黑衣人追击中提到她平地起雾、源石技艺能迷惑感知，她最终带伤逃离，并说自己不是被逼入绝境而是主动选择。",
        "supported_facts": [
            {"fact": "夜烟用源石技艺制造黑雾迷惑黑衣人的感知。",
             "evidence_refs": [
                 {"evidence_id": "activities/act15side/level_act15side_st04#chunk-0003",
                  "quote": "又平地起雾了？是她在施术"}]},
            {"fact": "黑衣人认为她的源石技艺能迷惑感知、难以抓住她。",
             "evidence_refs": [
                 {"evidence_id": "activities/act15side/level_act15side_st04#chunk-0003",
                  "quote": "不确定有没有杀伤力，但是能迷惑感知"}]},
        ],
    },
    # [18] 苦根如何推断苏茜被带到地下物流通道
    {
        "question": "苦根是如何推断苏茜被带到地下物流通道的？",
        "answer": "苦根推断贝希曼会把苏茜带到一个绝对安静、不被人察觉的地方，而他此前发现那家工厂下面有整个卡拉顿最大的老物流通道，这也解释了工厂设备去了哪里；于是他比对贝希曼伯爵的产业与老物流通道的交叉点，整理出七个最可能的地下通道入口，据此锁定苏茜被带到了地下物流通道。",
        "supported_facts": [
            {"fact": "苦根发现工厂下面有卡拉顿最大的老物流通道，并据此推断苏茜可能被带到那里。",
             "evidence_refs": [
                 {"evidence_id": "activities/act15side/level_act15side_st02#chunk-0001",
                  "quote": "我之前发现那家工厂下面有整个卡拉顿最大的老物流通道"}]},
            {"fact": "苦根比对贝希曼的产业与老物流通道的交叉点，整理出七个最可能的入口。",
             "evidence_refs": [
                 {"evidence_id": "activities/act15side/level_act15side_st02#chunk-0002",
                  "quote": "我刚才比对了一下贝希曼伯爵的产业和卡拉顿城区老物流通道的交叉点"}]},
        ],
    },
    # [19] 雷德是什么人/为何帮苏茜
    {
        "question": "雷德是什么人？他为什么帮助苏茜？",
        "answer": "雷德是附近工厂的一名乌萨斯移民工人，常到苏茜的店里、喜欢一个人看书，其他工人尊敬地叫他雷德老。他隐藏的身份是被称为红刀的人。他帮助苏茜是因为看不惯反对感染者的人频繁滋事，在苏茜被围堵、被追打时出手护送和解围。",
        "supported_facts": [
            {"fact": "雷德是附近工厂的乌萨斯移民工人，其他工人尊敬地称他雷德老。",
             "evidence_refs": [
                 {"evidence_id": "activities/act15side/level_act15side_st01#chunk-0002",
                  "quote": "他虽然是个乌萨斯移民，但是并不喜欢喝酒，而是喜欢在店里一个人看书"}]},
            {"fact": "雷德的真实身份是被称为红刀的人。",
             "evidence_refs": [
                 {"evidence_id": "activities/act15side/level_act15side_st02#chunk-0005",
                  "quote": "以前在乌萨斯的时候，他们都叫我“红刀”"}]},
        ],
    },
    # [20] 夏栎如何救出苏茜 —— 本池证据偏"救治/脱困后"，重写为可证内容
    {
        "question": "夏栎最终是如何救出苏茜的？",
        "answer": "根据现有证据可确认：苏茜脱困后由夏栎接手救治，夏栎为她静脉注射补充能量、用源石技艺控制炎症引发的高热、处理外伤，使她恢复稳定；证据还显示天火用源石技艺烧掉暴徒的源石炸弹将其制服。证据未直接给出夏栎把苏茜带出地下通道的完整过程。",
        "supported_facts": [
            {"fact": "夏栎为苏茜静脉注射补充能量、用源石技艺控制高热并处理外伤。",
             "evidence_refs": [
                 {"evidence_id": "activities/act10mini/level_act10mini_st05#chunk-0006",
                  "quote": "炎症引发的高热我可以用源石技艺暂时控制，避免留下后遗症"}]},
            {"fact": "天火用源石技艺烧掉暴徒的源石炸弹。",
             "evidence_refs": [
                 {"evidence_id": "activities/act10mini/level_act10mini_st03#chunk-0020",
                  "quote": "堆砌在墙角的源石炸弹的树脂外壳在火焰中熔化"}]},
        ],
    },
    # [21] 巫女林是什么样的组织/夜烟为何来自那里
    {
        "question": "巫女林是什么样的组织？夜烟为何来自那里？",
        "answer": "巫女林是一个收容感染者术师的组织，五年前因为拒绝和大公爵妥协而被维多利亚军队剿灭。夜烟是巫女林的第一百一十三位女巫，也是巫女林被灭后的幸存者，因此她来自巫女林、并以女巫的身份在各地漂泊。",
        "supported_facts": [
            {"fact": "巫女林是收容感染者术师的组织，因拒绝与大公爵妥协被维多利亚军队剿灭。",
             "evidence_refs": [
                 {"evidence_id": "activities/act15side/level_act15side_st02#chunk-0001",
                  "quote": "是那个收容感染者术师的组织"}]},
            {"fact": "夜烟是巫女林的第一百一十三位女巫，是巫女林的幸存者。",
             "evidence_refs": [
                 {"evidence_id": "activities/act15side/level_act15side_st04#chunk-0002",
                  "quote": "我是巫女林的第一百一十三位女巫，夜烟"}]},
        ],
    },
]
